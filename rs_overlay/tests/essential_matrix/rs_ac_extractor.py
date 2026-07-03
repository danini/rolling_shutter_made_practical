"""
rs_ac_extractor.py — Affine correspondence extractor for Rolling-Shutter testing.

Extracts local affine frames (LAFs) from image pairs using kornia's
KeyNetAffNetHardNet pipeline (pretrained, no manual weight download needed),
then converts them to the 12-column format required by estimateRSEssentialMatrix.

Alternatively, uses RoMa (dense matcher) to obtain a dense warp field and
derives affine transformations from the warp's local Jacobian via finite
differences.

Output format per row (all in normalized image coordinates, K^{-1} applied):
  [x1_n, y1_n, x2_n, y2_n, tau1, tau2, J00, J10, J20, J01, J11, J21]

where
  tau_i = y_i_pix / height - 0.5          (∈ [-0.5, 0.5])
  J_3x2 = [[J00, J01],                    (3×2 Jacobian dq2/dq1 in normalised coords)
            [J10, J11],
            [J20, J21]]                   (third row is always zeros)
  J_2x2_norm = diag(1/f2) @ A_pix @ diag(f1)
"""

import os
import tempfile
import numpy as np
import torch
import kornia
import kornia.feature as KF
from kornia.geometry.transform import warp_perspective


def _laf_to_numpy(lafs):
    """Convert (1, N, 2, 3) torch LAF tensor to (N, 2, 3) numpy array."""
    return lafs.squeeze(0).cpu().detach().numpy()


def _compute_affine_from_lafs(laf1_np, laf2_np):
    """
    Compute the 2x2 affine matrix A_pix (pixel coords) from a pair of LAFs.

    A LAF has shape (2, 3): columns are [col_axis, row_axis, center].
    The linear part is LAF[:, :2]  (a 2×2 matrix).
    A_pix = LAF2[:, :2] @ inv(LAF1[:, :2])
    """
    A = laf2_np[:, :2] @ np.linalg.inv(laf1_np[:, :2])
    return A  # (2, 2)


def extract_affine_correspondences(
    img1_np,
    img2_np,
    K1,
    K2,
    height,
    num_features=4096,
    ratio_threshold=0.9,
    device="cuda",
):
    """
    Extract affine correspondences from a grayscale image pair.

    Parameters
    ----------
    img1_np, img2_np : np.ndarray, shape (H, W), uint8
        Undistorted grayscale images.
    K1, K2 : np.ndarray, shape (3, 3)
        Camera intrinsics (for the undistorted images).
    height : int
        Image height in pixels (used to compute tau row coordinates).
    num_features : int
        Max number of keypoints per image.
    ratio_threshold : float
        Lowe ratio threshold for SMNN matching (default 0.9).
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    acs : np.ndarray, shape (N, 12), float64
        Affine correspondences in RS estimator format.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Convert to (1, 1, H, W) float tensors in [0, 1]
    def _to_tensor(img):
        t = torch.from_numpy(img.astype(np.float32) / 255.0)
        return t.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)

    img1_t = _to_tensor(img1_np)
    img2_t = _to_tensor(img2_np)

    # Detect keypoints and compute descriptors + LAFs
    detector = KF.KeyNetAffNetHardNet(
        num_features=num_features,
        upright=False,
        device=device,
    ).to(device)

    with torch.inference_mode():
        lafs1, resp1, desc1 = detector(img1_t)  # (1, N, 2, 3), (1, N, 1), (1, N, 128)
        lafs2, resp2, desc2 = detector(img2_t)

    # Match descriptors with soft mutual nearest-neighbour ratio test
    # match_smnn returns (distances (K,1), idxs (K,2))
    _, idxs = KF.match_smnn(
        desc1.squeeze(0),   # (N, 128)
        desc2.squeeze(0),   # (M, 128)
        th=ratio_threshold,
    )
    if idxs.shape[0] < 5:
        return np.zeros((0, 12), dtype=np.float64)

    idxs_np = idxs.cpu().numpy()  # (K, 2)
    lafs1_np = _laf_to_numpy(lafs1)  # (N, 2, 3)
    lafs2_np = _laf_to_numpy(lafs2)  # (M, 2, 3)

    # Intrinsics
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    fx2, fy2, cx2, cy2 = K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]

    n = idxs_np.shape[0]
    acs = np.zeros((n, 12), dtype=np.float64)

    for row, (i1, i2) in enumerate(idxs_np):
        laf1 = lafs1_np[i1]  # (2, 3)
        laf2 = lafs2_np[i2]  # (2, 3)

        # Pixel coordinates (center of LAF)
        x1_pix, y1_pix = laf1[0, 2], laf1[1, 2]
        x2_pix, y2_pix = laf2[0, 2], laf2[1, 2]

        # Normalise by K
        x1_n = (x1_pix - cx1) / fx1
        y1_n = (y1_pix - cy1) / fy1
        x2_n = (x2_pix - cx2) / fx2
        y2_n = (y2_pix - cy2) / fy2

        # Rolling-shutter row coordinate tau ∈ [-0.5, 0.5]
        tau1 = y1_pix / height - 0.5
        tau2 = y2_pix / height - 0.5

        # 2×2 affine in pixel coords
        A_pix = _compute_affine_from_lafs(laf1, laf2)  # (2, 2)

        # Convert to normalised coords:
        # q = K^{-1} p  =>  dq2/dq1 = diag(1/f2) @ A_pix @ diag(f1)
        J_2x2 = np.array([
            [A_pix[0, 0] * fx1 / fx2, A_pix[0, 1] * fy1 / fx2],
            [A_pix[1, 0] * fx1 / fy2, A_pix[1, 1] * fy1 / fy2],
        ])

        # Pad to 3×2: third row is zero (no depth change in affine model)
        J_3x2 = np.zeros((3, 2), dtype=np.float64)
        J_3x2[:2, :] = J_2x2

        # Pack into output: [x1n, y1n, x2n, y2n, tau1, tau2, J col-major flattened]
        # J col-major: J[0,0], J[1,0], J[2,0], J[0,1], J[1,1], J[2,1]
        acs[row, 0] = x1_n
        acs[row, 1] = y1_n
        acs[row, 2] = x2_n
        acs[row, 3] = y2_n
        acs[row, 4] = tau1
        acs[row, 5] = tau2
        acs[row, 6]  = J_3x2[0, 0]
        acs[row, 7]  = J_3x2[1, 0]
        acs[row, 8]  = J_3x2[2, 0]
        acs[row, 9]  = J_3x2[0, 1]
        acs[row, 10] = J_3x2[1, 1]
        acs[row, 11] = J_3x2[2, 1]

    return acs


def extract_affine_correspondences_roma(
    img1_np,
    img2_np,
    K1,
    K2,
    height,
    width,
    roma_model,
    num_samples=10000,
    device="cuda",
):
    """
    Extract affine correspondences using RoMa dense matching.

    Runs RoMa to get a dense warp field, then computes the local 2×2 affine
    (Jacobian) at each sampled feature via central finite differences on the
    warp.

    Parameters
    ----------
    img1_np, img2_np : np.ndarray, shape (H, W), uint8
        Undistorted grayscale images.
    K1, K2 : np.ndarray, shape (3, 3)
        Camera intrinsics (for the undistorted images).
    height, width : int
        Original image dimensions in pixels.
    roma_model : romatch model
        Pre-loaded RoMa model (e.g. from ``roma_outdoor(device)``).
    num_samples : int
        Number of correspondences to sample from the dense warp.
    device : str
        'cuda' or 'cpu'.

    Returns
    -------
    acs : np.ndarray, shape (N, 12), float64
        Affine correspondences in RS estimator format.
    """
    from PIL import Image

    # ── 1. Save undistorted images to temp files (RoMa upsample re-opens paths)
    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = os.path.join(tmpdir, "img1.png")
        path2 = os.path.join(tmpdir, "img2.png")
        Image.fromarray(img1_np).save(path1)
        Image.fromarray(img2_np).save(path2)

        # ── 2. Dense matching
        warp, certainty = roma_model.match(path1, path2, device=device)
        # warp: (H_r, W_r, 4) — [x1_norm, y1_norm, x2_norm, y2_norm] in [-1, 1]
        # certainty: (H_r, W_r)

    H_r, W_r = warp.shape[0], warp.shape[1]

    # ── 3. Convert to pixel coordinates (on CPU, numpy)
    warp_np = warp.cpu().numpy()  # (H_r, W_r, 4)
    x1_pix = (warp_np[:, :, 0] + 1.0) * width  / 2.0   # (H_r, W_r)
    y1_pix = (warp_np[:, :, 1] + 1.0) * height / 2.0
    x2_pix = (warp_np[:, :, 2] + 1.0) * width  / 2.0
    y2_pix = (warp_np[:, :, 3] + 1.0) * height / 2.0

    # ── 4. Dense Jacobian via central finite differences (vectorised)
    #   Grid step in original-image pixels between adjacent warp grid cells
    step_x = (x1_pix[0, 2] - x1_pix[0, 0]) / 2.0   # scalar
    step_y = (y1_pix[2, 0] - y1_pix[0, 0]) / 2.0   # scalar

    # dx2/dx1  (along columns, x-direction)
    A00 = (x2_pix[:, 2:] - x2_pix[:, :-2]) / (2.0 * step_x)   # (H_r, W_r-2)
    A10 = (y2_pix[:, 2:] - y2_pix[:, :-2]) / (2.0 * step_x)

    # dx2/dy1  (along rows, y-direction)
    A01 = (x2_pix[2:, :] - x2_pix[:-2, :]) / (2.0 * step_y)   # (H_r-2, W_r)
    A11 = (y2_pix[2:, :] - y2_pix[:-2, :]) / (2.0 * step_y)

    # Valid region: rows [1, H_r-2], cols [1, W_r-2]
    A00 = A00[1:-1, :]    # (H_r-2, W_r-2)
    A10 = A10[1:-1, :]
    A01 = A01[:, 1:-1]    # (H_r-2, W_r-2)
    A11 = A11[:, 1:-1]

    # ── 5. Sample sparse correspondences from the dense warp
    matches, cert = roma_model.sample(warp, certainty, num=num_samples)
    # matches: (K, 4) in [-1, 1] — [x1_n, y1_n, x2_n, y2_n]
    matches_np = matches.cpu().numpy()

    # ── 6. Convert sampled source coords to warp grid indices
    #   The warp grid spans [-1+1/W_r, 1-1/W_r] but round to nearest integer index
    j_idx = np.round((matches_np[:, 0] + 1.0) / 2.0 * (W_r - 1)).astype(int)
    i_idx = np.round((matches_np[:, 1] + 1.0) / 2.0 * (H_r - 1)).astype(int)

    # Clamp to valid region [1, H_r-2] / [1, W_r-2]
    i_idx = np.clip(i_idx, 1, H_r - 2)
    j_idx = np.clip(j_idx, 1, W_r - 2)

    # Offset by 1 to index into the valid-region arrays
    iv = i_idx - 1
    jv = j_idx - 1

    # ── 7. Look up Jacobian at each sample → A_pix (2×2)
    a00 = A00[iv, jv]   # (K,)
    a01 = A01[iv, jv]
    a10 = A10[iv, jv]
    a11 = A11[iv, jv]

    # ── 8. Pixel coordinates of sampled points
    s_x1_pix = (matches_np[:, 0] + 1.0) * width  / 2.0
    s_y1_pix = (matches_np[:, 1] + 1.0) * height / 2.0
    s_x2_pix = (matches_np[:, 2] + 1.0) * width  / 2.0
    s_y2_pix = (matches_np[:, 3] + 1.0) * height / 2.0

    # Intrinsics
    fx1, fy1, cx1, cy1 = K1[0, 0], K1[1, 1], K1[0, 2], K1[1, 2]
    fx2, fy2, cx2, cy2 = K2[0, 0], K2[1, 1], K2[0, 2], K2[1, 2]

    # Normalise by K
    x1_n = (s_x1_pix - cx1) / fx1
    y1_n = (s_y1_pix - cy1) / fy1
    x2_n = (s_x2_pix - cx2) / fx2
    y2_n = (s_y2_pix - cy2) / fy2

    # ── 9. Rolling-shutter row coordinate tau ∈ [-0.5, 0.5]
    tau1 = s_y1_pix / height - 0.5
    tau2 = s_y2_pix / height - 0.5

    # ── 10. Convert A_pix to normalised Jacobian: J = diag(1/f2) @ A_pix @ diag(f1)
    J00 = a00 * fx1 / fx2
    J01 = a01 * fy1 / fx2
    J10 = a10 * fx1 / fy2
    J11 = a11 * fy1 / fy2

    # ── 11. Pack into 12-column format
    #   [x1n, y1n, x2n, y2n, tau1, tau2, J00, J10, J20, J01, J11, J21]
    #   J col-major: J[0,0], J[1,0], J[2,0], J[0,1], J[1,1], J[2,1]
    K_pts = matches_np.shape[0]
    acs = np.zeros((K_pts, 12), dtype=np.float64)
    acs[:, 0]  = x1_n
    acs[:, 1]  = y1_n
    acs[:, 2]  = x2_n
    acs[:, 3]  = y2_n
    acs[:, 4]  = tau1
    acs[:, 5]  = tau2
    acs[:, 6]  = J00       # J[0,0]
    acs[:, 7]  = J10       # J[1,0]
    acs[:, 8]  = 0.0       # J[2,0]  (third row always zero)
    acs[:, 9]  = J01       # J[0,1]
    acs[:, 10] = J11       # J[1,1]
    acs[:, 11] = 0.0       # J[2,1]

    return acs

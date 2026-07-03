"""
tester_rs_superansac.py — Benchmark RS-aware vs GS essential-matrix estimators
                          on the TUM Rolling-Shutter dataset.

Parallelism strategy:
  Phase 1 (sequential): AC extraction per pair on GPU.
  Phase 2 (parallel):   RANSAC estimation on CPU via ThreadPoolExecutor.

Usage:
    python tester_rs_superansac.py \
        --sequences 1 2 3 --strides 1 5 --max_pairs 200 \
        --methods rs_ac gs_5pt --scoring MAGSAC \
        --threshold 1.5 --max_iterations 1000 \
        --workers 8 \
        --output /tmp/results_rs.csv

Output CSV columns:
    method, seq, stride, pair_id,
    R_err_deg, t_err_deg, omega_err, v_err,
    n_acs, n_inliers, time_ms,
    sampler, scoring, lo, threshold, max_iter

omega_err = ||omega1_est - omega1_gt|| + ||omega2_est - omega2_gt||  (norm-based)
v_err     = ||v1_est - v1_gt|| + ||v2_est - v2_gt||                 (norm-based)
"""

import argparse
import csv
import hashlib
import json
import tqdm
import os
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")

import cv2
import h5py
import numpy as np

# ── project paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR   = os.path.dirname(SCRIPT_DIR)
SUPERANSAC  = os.path.dirname(TESTS_DIR)
ROOT        = os.path.dirname(SUPERANSAC)

sys.path.insert(0, TESTS_DIR)               # for evaluation.py, functions.py
sys.path.insert(0, SUPERANSAC)              # for pysuperansac.so
sys.path.insert(0, ROOT)                    # for third_party/

from evaluation import evaluate_R_t, pose_auc

# Lazy imports: torch/kornia/romatch only needed when there is no AC cache on disk.
# This avoids a hard dependency on GPU packages when ACs are already cached.
extract_affine_correspondences = None
extract_affine_correspondences_roma = None
_roma_model = None

import pysuperansac

# TUM-RS dataset loader lives in tests/datasets/
sys.path.insert(0, os.path.join(TESTS_DIR, "datasets"))
from tum_rs import TumRS
from euroc_mav import EuRoCMAV


# ── AC cache (HDF5) ────────────────────────────────────────────────────────────

def _cache_fingerprint(args):
    """Stable string key encoding all config that affects AC extraction."""
    cfg = dict(
        sequences=sorted(args.sequences) if args.sequences else "all",
        strides=sorted(args.strides),
        max_pairs=args.max_pairs,
        features=args.features,
        num_features=args.num_features,
        ratio_threshold=args.ratio_threshold,
        roma_samples=args.roma_samples,
        dataset_root=args.dataset_root,
    )
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def load_ac_cache(cache_path, args, n_pairs):
    """
    Try to load ACs from an HDF5 cache file.

    Returns list of np.ndarray (length n_pairs, may contain empty arrays)
    if the cache is valid for the current config, else None.
    """
    if not os.path.isfile(cache_path):
        return None
    try:
        with h5py.File(cache_path, "r") as f:
            if f.attrs.get("fingerprint") != _cache_fingerprint(args):
                print("  [cache] config changed — ignoring existing cache.", flush=True)
                return None
            if f.attrs.get("n_pairs") != n_pairs:
                print("  [cache] pair count mismatch — ignoring existing cache.", flush=True)
                return None
            # Check all pairs are present
            if not all(str(i) in f["acs"] for i in range(n_pairs)):
                print("  [cache] incomplete cache — will re-extract missing pairs.", flush=True)
                # Return partial cache; missing indices will be None
                acs_list = []
                for i in range(n_pairs):
                    if str(i) in f["acs"]:
                        acs_list.append(f["acs"][str(i)][:])
                    else:
                        acs_list.append(None)
                return acs_list
            acs_list = [f["acs"][str(i)][:] for i in range(n_pairs)]
        print(f"  [cache] loaded {n_pairs} pairs from {cache_path}", flush=True)
        return acs_list
    except Exception as e:
        print(f"  [cache] failed to load ({e}) — re-extracting.", flush=True)
        return None


def save_ac_cache(cache_path, acs_list, args):
    """Save extracted ACs to an HDF5 cache file."""
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with h5py.File(cache_path, "w") as f:
        f.attrs["fingerprint"] = _cache_fingerprint(args)
        f.attrs["n_pairs"] = len(acs_list)
        grp = f.create_group("acs")
        for i, acs in enumerate(acs_list):
            grp.create_dataset(str(i), data=acs, compression="lzf")
    print(f"  [cache] saved {len(acs_list)} pairs to {cache_path}", flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def angle_between_vectors(a, b):
    """Angular error in degrees between two direction vectors."""
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    cos = np.clip(np.dot(a, b), -1.0, 1.0)
    return np.degrees(np.arccos(abs(cos)))


def build_config(args, threshold):
    """Build a pysuperansac.RANSACSettings from CLI args.

    Parameters
    ----------
    threshold : float
        Inlier threshold in the units expected by the estimator:
          - RS estimator : normalised image coordinates (pixels / fy)
          - GS estimator : pixels (normalised internally by focal length)
    """
    cfg = pysuperansac.RANSACSettings()
    cfg.inlier_threshold  = threshold
    cfg.min_iterations    = args.min_iterations
    cfg.max_iterations    = args.max_iterations
    cfg.confidence        = 0.99

    sampler_map = {
        "Uniform":  pysuperansac.SamplerType.Uniform,
        "PROSAC":   pysuperansac.SamplerType.PROSAC,
        "PNAPSAC":  pysuperansac.SamplerType.ProgressiveNAPSAC,
    }
    scoring_map = {
        "RANSAC":  pysuperansac.ScoringType.RANSAC,
        "MSAC":    pysuperansac.ScoringType.MSAC,
        "MAGSAC":  pysuperansac.ScoringType.MAGSAC,
        "ACRANSAC":pysuperansac.ScoringType.ACRANSAC,
    }
    lo_map = {
        "Nothing":       pysuperansac.LocalOptimizationType.Nothing,
        "LSQ":           pysuperansac.LocalOptimizationType.LSQ,
        "NestedRANSAC":  pysuperansac.LocalOptimizationType.NestedRANSAC,
    }
    fo_map = {
        "Nothing":       pysuperansac.LocalOptimizationType.Nothing,
        "LSQ":           pysuperansac.LocalOptimizationType.LSQ,
        "NestedRANSAC":  pysuperansac.LocalOptimizationType.NestedRANSAC,
    }

    cfg.sampler            = sampler_map.get(args.sampler, pysuperansac.SamplerType.PROSAC)
    cfg.scoring            = scoring_map.get(args.scoring, pysuperansac.ScoringType.MAGSAC)
    cfg.local_optimization = lo_map.get(args.lo, pysuperansac.LocalOptimizationType.LSQ)
    cfg.final_optimization = lo_map.get(args.fo, pysuperansac.LocalOptimizationType.LSQ)

    # Cap local/final optimiser iterations to prevent slow or "stuck" runs.
    # Default is 20; 3 rounds of BA refinement retain almost all accuracy.
    cfg.local_optimization_settings.max_iterations = 3
    cfg.final_optimization_settings.max_iterations  = 3

    # Only refine the single best model after RANSAC (not top-3).
    # With large inlier sets (N>1000), each BA call can be 300ms+;
    # running 3 is the primary cause of "stuck" behaviour.
    cfg.local_opt_k = 1
    return cfg


def run_rs_ac(acs_12col, fy_over_h, image_sizes, cfg):
    """Run estimateRSEssentialMatrix; return (model_8x3, inliers, time_ms)."""
    if acs_12col.shape[0] < 5:
        return None, [], 0.0

    probs = np.ones(acs_12col.shape[0], dtype=np.float64)
    tic = time.perf_counter()
    model, inliers, _, _ = pysuperansac.estimateRSEssentialMatrix(
        np.ascontiguousarray(acs_12col, dtype=np.float64),
        fy_over_h,
        np.array(image_sizes, dtype=np.float64),
        probs,
        config=cfg,
    )
    toc = time.perf_counter()
    return model, list(inliers), (toc - tic) * 1e3


def run_rs_ac_7direct(acs_12col, fy_over_h, image_sizes, cfg,
                      ac_weight=0.0, nonmin_solver="pc"):
    """Run estimateRSEssentialMatrix7AC; return (model_8x3, inliers, time_ms)."""
    if acs_12col.shape[0] < 7:
        return None, [], 0.0

    probs = np.ones(acs_12col.shape[0], dtype=np.float64)
    tic = time.perf_counter()
    model, inliers, _, _ = pysuperansac.estimateRSEssentialMatrix7AC(
        np.ascontiguousarray(acs_12col, dtype=np.float64),
        fy_over_h,
        np.array(image_sizes, dtype=np.float64),
        probs,
        config=cfg,
        ac_weight=ac_weight,
        nonmin_solver=nonmin_solver,
    )
    toc = time.perf_counter()
    return model, list(inliers), (toc - tic) * 1e3


def _norm_to_pix(acs_12col, K1, K2):
    """Convert normalised image coords (cols 0-3 of ACs) to pixel coords."""
    pts1_pix = (acs_12col[:, :2] * np.array([K1[0, 0], K1[1, 1]])
                + np.array([K1[0, 2], K1[1, 2]]))
    pts2_pix = (acs_12col[:, 2:4] * np.array([K2[0, 0], K2[1, 1]])
                + np.array([K2[0, 2], K2[1, 2]]))
    return np.column_stack([pts1_pix, pts2_pix])


def run_gs_5pt(acs_12col, K1, K2, image_sizes_wh, cfg_gs):
    """
    Run estimateEssentialMatrix (GS 5pt baseline) from the first 4 cols of ACs.
    Returns (E_est, R_est, t_est, elapsed_ms, inliers).
    """
    if acs_12col.shape[0] < 5:
        return None, None, None, 0.0, []

    matches_pix = _norm_to_pix(acs_12col, K1, K2)
    probs = np.ones(acs_12col.shape[0], dtype=np.float64)

    tic = time.perf_counter()
    E_est, inliers, _, _ = pysuperansac.estimateEssentialMatrix(
        np.ascontiguousarray(matches_pix, dtype=np.float64),
        K1,
        K2,
        np.array(image_sizes_wh, dtype=np.float64),
        probs,
        config=cfg_gs,
    )
    toc = time.perf_counter()
    elapsed_ms = (toc - tic) * 1e3

    if E_est is None or len(inliers) == 0:
        return E_est, None, None, elapsed_ms, list(inliers)

    # Decompose E → R, t using inlier normalised coordinates
    norm1 = acs_12col[list(inliers), :2]
    norm2 = acs_12col[list(inliers), 2:4]
    _, R, t, _ = cv2.recoverPose(E_est, norm1, norm2)
    return E_est, R, t[:, 0], elapsed_ms, list(inliers)


def run_rs_baseline(acs_12col, method, fy_over_h, image_sizes, cfg):
    """Run estimateRSEssentialMatrixGeneric for a baseline method.

    Baseline methods: dai_20pt, dai_44pt (Dai et al. CVPR 2016).
    Returns (model_8x3, inliers, time_ms).
    """
    min_pts = {"dai_20pt": 20, "dai_44pt": 44}.get(method, 20)
    if acs_12col.shape[0] < min_pts:
        return None, [], 0.0

    probs = np.ones(acs_12col.shape[0], dtype=np.float64)
    tic = time.perf_counter()
    model, inliers, _, _ = pysuperansac.estimateRSEssentialMatrixGeneric(
        np.ascontiguousarray(acs_12col, dtype=np.float64),
        method,
        fy_over_h,
        np.array(image_sizes, dtype=np.float64),
        probs,
        config=cfg,
    )
    toc = time.perf_counter()
    return model, list(inliers), (toc - tic) * 1e3


def extract_pose_from_rs_model(model_8x3):
    """
    Extract R, t, and RS params from an 8×3 RS model matrix.

    Row layout: R(0:3), t^T(3), omega1^T(4), v1^T(5), omega2^T(6), v2^T(7)
    """
    R      = model_8x3[:3, :]        # (3, 3)
    t      = model_8x3[3, :]         # (3,)
    omega1 = model_8x3[4, :]         # (3,)
    v1     = model_8x3[5, :]         # (3,)
    omega2 = model_8x3[6, :]         # (3,)
    v2     = model_8x3[7, :]         # (3,)
    return R, t, omega1, v1, omega2, v2


# ── parallel worker ────────────────────────────────────────────────────────────

def process_pair_ransac(pair_idx, data, acs, args):
    """
    Run all requested RANSAC methods for a single pre-extracted pair.

    Called from worker threads (Phase 2).  Builds its own RANSACSettings
    objects so there is no shared mutable state between threads.

    Returns
    -------
    rows : list[dict]   — one CSV row per method
    errors : list[(method, R_err, t_err)]
    """
    seq    = data["seq"]
    stride = data["stride"]
    K1     = data["K1"]
    K2     = data["K2"]
    R_gt   = data["R_1_2"]
    t_gt   = data["T_1_2"]
    omega1_gt = data.get("omega1_gt", None)
    v1_gt     = data.get("v1_gt",    None)
    omega2_gt = data.get("omega2_gt", None)
    v2_gt     = data.get("v2_gt",    None)
    H, W   = data["img1_np"].shape[:2]

    fy_over_h  = K1[1, 1] / H
    image_sizes = [float(W), float(H), float(W), float(H)]
    n_acs = acs.shape[0]

    # Normalise the shared pixel threshold for each estimator:
    #   RS estimator uses Sampson distance in normalised image coords  → divide by fy
    #   GS estimator normalises points by K internally                 → pass pixels
    fy = K1[1, 1]
    fx = K1[0, 0]
    thr_normalizer = 0.5 * (fx + fy)  # average focal length in pixels
    rs_threshold = args.threshold / thr_normalizer   # normalised coords
    gs_threshold = args.threshold        # pixels

    # Each thread gets its own config objects (pysuperansac objects are not
    # guaranteed thread-safe to share if they hold mutable internal state)
    cfg    = build_config(args, threshold=rs_threshold)
    cfg_gs = build_config(args, threshold=gs_threshold)

    base_row = dict(
        seq=seq, stride=stride, pair_id=pair_idx,
        n_acs=n_acs,
        sampler=args.sampler, scoring=args.scoring, lo=args.lo,
        threshold=args.threshold, max_iter=args.max_iterations,
    )

    rows   = []
    errors = []  # (method, R_err, t_err, n_inliers, time_ms, omega_err, v_err)

    for method in args.methods:
        row = dict(base_row, method=method)

        if n_acs < 5:
            row.update(R_err_deg=180, t_err_deg=180,
                       omega_err=180, v_err=180,
                       n_inliers=0, time_ms=0)
            rows.append(row)
            errors.append((method, 180.0, 180.0, 0, 0, float("nan"), float("nan")))
            continue

        try:
            if method == "rs_ac":
                model, inliers, time_ms = run_rs_ac(acs, fy_over_h, image_sizes, cfg)
                n_inliers = len(inliers)

                if model is None or n_inliers < 5:
                    R_err, t_err = 180.0, 180.0
                    omega_err, v_err = float("nan"), float("nan")
                else:
                    R_est, t_est, omega1_est, v1_est, omega2_est, v2_est = extract_pose_from_rs_model(model)
                    errs = evaluate_R_t(R_gt, t_gt, R_est, t_est)
                    R_err, t_err = errs[0], errs[1]
                    omega_err = (np.linalg.norm(omega1_est - omega1_gt) + np.linalg.norm(omega2_est - omega2_gt)
                                 if omega1_gt is not None else float("nan"))
                    v_err = (np.linalg.norm(v1_est - v1_gt) + np.linalg.norm(v2_est - v2_gt)
                             if v1_gt is not None else float("nan"))

            elif method == "ac_rs_7direct":
                model, inliers, time_ms = run_rs_ac_7direct(
                    acs, fy_over_h, image_sizes, cfg, args.ac_weight, args.nonmin_solver)
                n_inliers = len(inliers)

                if model is None or n_inliers < 7:
                    R_err, t_err = 180.0, 180.0
                    omega_err, v_err = float("nan"), float("nan")
                else:
                    R_est, t_est, omega1_est, v1_est, omega2_est, v2_est = extract_pose_from_rs_model(model)
                    errs = evaluate_R_t(R_gt, t_gt, R_est, t_est)
                    R_err, t_err = errs[0], errs[1]
                    omega_err = (np.linalg.norm(omega1_est - omega1_gt) + np.linalg.norm(omega2_est - omega2_gt)
                                 if omega1_gt is not None else float("nan"))
                    v_err = (np.linalg.norm(v1_est - v1_gt) + np.linalg.norm(v2_est - v2_gt)
                             if v1_gt is not None else float("nan"))

            elif method in ("dai_20pt", "dai_44pt"):
                model, inliers, time_ms = run_rs_baseline(
                    acs, method, fy_over_h, image_sizes, cfg)
                n_inliers = len(inliers)

                if model is None or n_inliers < 5:
                    R_err, t_err = 180.0, 180.0
                    omega_err, v_err = float("nan"), float("nan")
                else:
                    R_est, t_est, omega1_est, v1_est, omega2_est, v2_est = extract_pose_from_rs_model(model)
                    errs = evaluate_R_t(R_gt, t_gt, R_est, t_est)
                    R_err, t_err = errs[0], errs[1]
                    omega_err = (np.linalg.norm(omega1_est - omega1_gt) + np.linalg.norm(omega2_est - omega2_gt)
                                 if omega1_gt is not None else float("nan"))
                    v_err = (np.linalg.norm(v1_est - v1_gt) + np.linalg.norm(v2_est - v2_gt)
                             if v1_gt is not None else float("nan"))

            elif method == "gs_5pt":
                _, R_est, t_est, time_ms, inliers = run_gs_5pt(
                    acs, K1, K2, image_sizes, cfg_gs)
                n_inliers = len(inliers)

                if R_est is None or n_inliers < 5:
                    R_err, t_err = 180.0, 180.0
                else:
                    errs = evaluate_R_t(R_gt, t_gt, R_est,
                                        t_est if t_est is not None else np.zeros(3))
                    R_err, t_err = errs[0], errs[1]
                omega_err, v_err = float("nan"), float("nan")

        except Exception as e:
            print(f"  [pair {pair_idx}, {method}] estimation failed: {e}")
            R_err, t_err = 180.0, 180.0
            omega_err, v_err = float("nan"), float("nan")
            n_inliers = 0
            time_ms = 0.0

        row.update(
            R_err_deg=round(float(R_err), 4),
            t_err_deg=round(float(t_err), 4),
            omega_err=round(float(omega_err), 4) if not np.isnan(omega_err) else "",
            v_err=round(float(v_err), 4) if not np.isnan(v_err) else "",
            n_inliers=n_inliers,
            time_ms=round(float(time_ms), 2),
        )
        rows.append(row)
        errors.append((method, float(R_err), float(t_err), n_inliers, float(time_ms), float(omega_err), float(v_err)))

    return rows, errors


# ── main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RS-aware vs GS benchmark on TUM-RS dataset")
    parser.add_argument("--dataset", type=str, default="tum_rs",
                        choices=["tum_rs", "euroc"],
                        help="Dataset to evaluate on")
    parser.add_argument("--dataset_root", type=str,
                        default=None,
                        help="Dataset root dir (auto-set per dataset if omitted)")
    parser.add_argument("--sequences", type=int, nargs="+",
                        default=None,
                        help="Sequence indices (TUM-RS: 1-10) or omit for all")
    parser.add_argument("--strides", type=int, nargs="+", default=None,
                        help="Frame strides (default: [1,5] for TUM-RS, [10,20] for EuRoC)")
    parser.add_argument("--max_pairs", type=int, default=200)
    ALL_METHODS = [
        "rs_ac", "ac_rs_7direct",              # our RS methods
        "dai_20pt", "dai_44pt",                # Dai et al. (CVPR 2016) RS baselines
        "gs_5pt",                              # GS baseline (5 PCs)
    ]
    parser.add_argument("--methods", type=str, nargs="+",
                        choices=ALL_METHODS,
                        default=["rs_ac", "ac_rs_7direct", "dai_20pt", "dai_44pt", "gs_5pt"])
    parser.add_argument("--features", type=str, default="KeyNetAffNetHardNet",
                        choices=["KeyNetAffNetHardNet", "RoMa"],
                        help="Feature extractor for affine correspondences")
    parser.add_argument("--num_features", type=int, default=4096,
                        help="Max keypoints per image for KeyNetAffNetHardNet")
    parser.add_argument("--ratio_threshold", type=float, default=0.9,
                        help="SMNN ratio threshold for matching")
    parser.add_argument("--roma_samples", type=int, default=500,
                        help="Number of correspondences to sample from RoMa warp")
    parser.add_argument("--sampler", type=str, default="PROSAC",
                        choices=["Uniform", "PROSAC", "PNAPSAC"])
    parser.add_argument("--scoring", type=str, default="MSAC",
                        choices=["RANSAC", "MSAC", "MAGSAC", "ACRANSAC"])
    parser.add_argument("--lo", type=str, default="NestedRANSAC",
                        choices=["Nothing", "LSQ", "NestedRANSAC"])
    parser.add_argument("--fo", type=str, default="LSQ",
                        choices=["Nothing", "LSQ", "NestedRANSAC"])
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Inlier threshold in pixels (default 1.5). "
                             "Converted to normalised coords (÷ fy) for the RS estimator; "
                             "passed directly as pixels to the GS estimator.")
    parser.add_argument("--min_iterations", type=int, default=50)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--ac_weight", type=float, default=0.0,
                        help="Weight for affine residuals in the LM refinement "
                             "(0 = epipolar-only, >0 = also minimise AC residuals)")
    parser.add_argument("--nonmin_solver", type=str, default="pc",
                        choices=["pc", "ba"],
                        help="Non-minimal solver for 7AC: "
                             "'pc' = PCRSRefinementSolver (fast, safe for large N), "
                             "'ba' = RSEssentialMatrixBundleAdjustmentSolver (AC-based)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel RANSAC worker threads "
                             "(default: all CPU cores)")
    parser.add_argument("--ac_cache", type=str,
                        default=None,
                        help="Path to HDF5 cache file for extracted ACs. "
                             "If the file exists and the config matches, "
                             "Phase 1 is skipped entirely. "
                             "Auto-generated if not specified.")
    parser.add_argument("--output", type=str,
                        default="/tmp/results_rs_superansac.csv")
    args = parser.parse_args()

    # Set dataset-specific defaults
    if args.dataset_root is None:
        args.dataset_root = {
            "tum_rs": "/media/hdd3tb/datasets/tum_rs",
            "euroc":  "/media/hdd3tb/datasets/euroc_mav",
        }[args.dataset]
    if args.strides is None:
        args.strides = {
            "tum_rs": [1, 5],
            "euroc":  [10, 20],
        }[args.dataset]
    if args.sequences is None and args.dataset == "tum_rs":
        args.sequences = list(range(1, 11))

    # Auto-generate cache path encoding the extraction config so different
    # parameter combos get separate cache files automatically.
    if args.ac_cache is None:
        feat_tag = {"KeyNetAffNetHardNet": "KNAH", "RoMa": "RoMa"}[args.features]
        ds_tag = args.dataset
        if args.sequences is not None:
            seqs_tag = "-".join(str(s) for s in sorted(args.sequences))
        else:
            seqs_tag = "all"
        str_tag  = "-".join(str(s) for s in sorted(args.strides))
        parts = [
            f"ac_cache_{ds_tag}_{feat_tag}",
            f"seq{seqs_tag}",
            f"str{str_tag}",
            f"p{args.max_pairs}",
            f"f{args.num_features}",
            f"r{int(args.ratio_threshold * 100)}",
        ]
        if args.features == "RoMa":
            parts.append(f"rs{args.roma_samples}")
        out_dir = os.path.dirname(os.path.abspath(args.output))
        args.ac_cache = os.path.join(out_dir, "_".join(parts) + ".h5")

    # ── dataset ───────────────────────────────────────────────────────────────
    if args.dataset == "tum_rs":
        print(f"Loading TUM-RS: seqs={args.sequences}, strides={args.strides}, "
              f"max_pairs={args.max_pairs}")
        dataset = TumRS(
            root_dir=args.dataset_root,
            sequences=args.sequences,
            strides=args.strides,
            max_pairs=args.max_pairs,
        )
    elif args.dataset == "euroc":
        print(f"Loading EuRoC MAV: strides={args.strides}, "
              f"max_pairs={args.max_pairs}")
        dataset = EuRoCMAV(
            root_dir=args.dataset_root,
            sequences=None,  # auto-detect
            strides=args.strides,
            max_pairs=args.max_pairs,
        )
    n_pairs = len(dataset)
    print(f"  {n_pairs} pairs total | {args.workers} RANSAC workers")

    # ── output CSV ───────────────────────────────────────────────────────────
    fieldnames = [
        "method", "seq", "stride", "pair_id",
        "R_err_deg", "t_err_deg", "omega_err", "v_err",
        "n_acs", "n_inliers", "time_ms",
        "sampler", "scoring", "lo", "threshold", "max_iter",
    ]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    csvfile = open(args.output, "w", newline="")
    writer  = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csv_lock = threading.Lock()

    # ── Phase 1: AC extraction (sequential, GPU) — with HDF5 caching ────────────
    cached_acs = load_ac_cache(args.ac_cache, args, n_pairs)
    need_extraction = cached_acs is None or any(a is None for a in (cached_acs or []))

    if need_extraction:
        print(f"\nPhase 1: extracting ACs on {args.device} "
              f"(features={args.features}, sequential)...", flush=True)
        if cached_acs is None:
            cached_acs = [None] * n_pairs
    else:
        print(f"\nPhase 1: all ACs loaded from cache ({args.ac_cache})", flush=True)

    pair_list = []  # [(pair_idx, slim_data_dict, acs_array)]
    newly_extracted = 0
    for idx in range(n_pairs):
        data = dataset[idx]
        H, W = data["img1_np"].shape[:2]

        if cached_acs[idx] is not None:
            acs = cached_acs[idx]
        else:
            if args.features == "KeyNetAffNetHardNet":
                global extract_affine_correspondences
                if extract_affine_correspondences is None:
                    from rs_ac_extractor import extract_affine_correspondences
                try:
                    acs = extract_affine_correspondences(
                        data["img1_np"], data["img2_np"],
                        data["K1"], data["K2"],
                        height=H,
                        num_features=args.num_features,
                        ratio_threshold=args.ratio_threshold,
                        device=args.device,
                    )
                except Exception as e:
                    print(f"  [pair {idx}] AC extraction failed: {e}", flush=True)
                    acs = np.zeros((0, 12))

            elif args.features == "RoMa":
                global extract_affine_correspondences_roma, _roma_model
                if extract_affine_correspondences_roma is None:
                    from rs_ac_extractor import extract_affine_correspondences_roma
                if _roma_model is None:
                    from romatch import roma_outdoor
                    print("  Loading RoMa model...", flush=True)
                    _roma_model = roma_outdoor(device=args.device)
                try:
                    acs = extract_affine_correspondences_roma(
                        data["img1_np"], data["img2_np"],
                        data["K1"], data["K2"],
                        height=H, width=W,
                        roma_model=_roma_model,
                        num_samples=args.roma_samples,
                        device=args.device,
                    )
                except Exception as e:
                    print(f"  [pair {idx}] RoMa AC extraction failed: {e}",
                          flush=True)
                    acs = np.zeros((0, 12))

            cached_acs[idx] = acs
            newly_extracted += 1
            if newly_extracted % 20 == 0:
                print(f"  [{idx+1}/{n_pairs}] ACs extracted  "
                      f"(last: {acs.shape[0]} correspondences)", flush=True)

        # Keep only the metadata needed for RANSAC — drop images to save RAM
        slim = {
            "seq":       data["seq"],
            "stride":    data["stride"],
            "K1":        data["K1"],
            "K2":        data["K2"],
            "R_1_2":     data["R_1_2"],
            "T_1_2":     data["T_1_2"],
            "omega1_gt": data.get("omega1_gt"),
            "v1_gt":     data.get("v1_gt"),
            "omega2_gt": data.get("omega2_gt"),
            "v2_gt":     data.get("v2_gt"),
            "img1_np":   np.empty((H, W), dtype=np.uint8),  # shape only, no content
        }
        pair_list.append((idx, slim, acs))

    if newly_extracted > 0:
        print(f"  extracted {newly_extracted} new pairs", flush=True)
        save_ac_cache(args.ac_cache, cached_acs, args)
    del cached_acs  # free memory before Phase 2



    # ── Phase 2: RANSAC (parallel, CPU threads) ───────────────────────────────
    print(f"\nPhase 2: running RANSAC with {args.workers} threads...", flush=True)
    all_errors   = {m: [] for m in args.methods}
    all_inliers  = {m: [] for m in args.methods}
    all_times    = {m: [] for m in args.methods}
    all_omega    = {m: [] for m in args.methods}
    all_v        = {m: [] for m in args.methods}
    errors_lock  = threading.Lock()
    completed    = 0
    completed_lock = threading.Lock()

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {
                executor.submit(process_pair_ransac, idx, data, acs, args): idx
                for idx, data, acs in pair_list
            }

            for future in tqdm.tqdm(as_completed(future_to_idx), total=len(future_to_idx)):
                pair_idx = future_to_idx[future]
                try:
                    rows, errors = future.result()
                except Exception as e:
                    print(f"  [pair {pair_idx}] worker crashed: {e}")
                    rows, errors = [], []

                with csv_lock:
                    for row in rows:
                        writer.writerow(row)
                    csvfile.flush()

                with errors_lock:
                    for method, r_err, t_err, inlier_num, time_ms, w_err, ve_err in errors:
                        all_errors[method].append((r_err, t_err))
                        all_inliers[method].append(inlier_num)
                        all_times[method].append(time_ms)
                        all_omega[method].append(w_err)
                        all_v[method].append(ve_err)

                with completed_lock:
                    completed += 1
                    done = completed
    else:
        for idx, data, acs in tqdm.tqdm(pair_list):
            rows, errors = process_pair_ransac(idx, data, acs, args)

            for row in rows:
                writer.writerow(row)
            csvfile.flush()

            for method, r_err, t_err, inlier_num, time_ms, w_err, ve_err in errors:
                all_errors[method].append((r_err, t_err))
                all_inliers[method].append(inlier_num)
                all_times[method].append(time_ms)
                all_omega[method].append(w_err)
                all_v[method].append(ve_err)

    csvfile.close()

    # ── AUC summary ───────────────────────────────────────────────────────────
    print("\n=== AUC summary (pose error = max(R_err, t_err) in degrees) ===")
    thresholds = [5, 10, 20]
    for method in args.methods:
        errs = all_errors[method]
        pose_errs = [max(r, t) for r, t in errs]
        aucs = pose_auc(pose_errs, thresholds)
        auc_str = ", ".join(f"AUC@{t}°={v:.3f}" for t, v in zip(thresholds, aucs))

        # Median omega/v errors (excluding NaN for GS methods)
        w_valid = [x for x in all_omega[method] if not np.isnan(x)]
        v_valid = [x for x in all_v[method] if not np.isnan(x)]
        w_med = f"{np.median(w_valid):.4f}" if w_valid else "N/A"
        v_med = f"{np.median(v_valid):.4f}" if v_valid else "N/A"

        print(f"  {method:14s}: {auc_str}  "
              f"med_ω={w_med}, med_v={v_med}  "
              f"(N={len(errs)}, avg time={np.mean(all_times[method]):.1f}ms)")

    print(f"\nResults written to: {args.output}")


if __name__ == "__main__":
    main()

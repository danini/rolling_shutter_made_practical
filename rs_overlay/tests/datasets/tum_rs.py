"""
TUM Rolling-Shutter VIO Dataset loader for relative pose evaluation.

Dataset layout:
    root_dir/
        euroc/
            dataset-seq{N}/
                mav0/
                    cam0/data/*.png   (rolling-shutter camera)
                    cam0/data.csv     (timestamp → filename)
                    cam1/data/*.png   (global-shutter camera)
                    cam1/data.csv
                    mocap0/data.csv   (ground-truth poses at ~120 Hz)
        poses/
            dataset-seq{N}_poses_cam0.txt  (TUM format poses)
        calibration/
            camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml

GT pose format in mocap0/data.csv:
    #timestamp [ns], p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m],
    q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []

Reference: Schubert et al., "The TUM VI Benchmark for Evaluating
Visual-Inertial Odometry", IROS 2018 / "Rolling-Shutter Modelling
for Direct Visual-Inertial Odometry".
"""

import os
import cv2
import yaml
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp
try:
    from torch.utils.data import Dataset as _TorchDataset, DataLoader
    import torch
    _HAS_TORCH = True
except ImportError:
    _TorchDataset = object
    DataLoader = None
    torch = None
    _HAS_TORCH = False


# Row-readout time for the TUM-RS camera (29.47 µs/row × 1024 rows)
TUM_RS_READOUT_TIME_S = 29.47e-6 * 1024   # ≈ 30.18 ms per frame


def simple_collate_fn(sample):
    return sample


def _quat_to_rotmat(qw, qx, qy, qz):
    """Convert wxyz quaternion to 3×3 rotation matrix."""
    return Rotation.from_quat([qx, qy, qz, qw]).as_matrix()


def _load_mocap(mocap_csv: str):
    """
    Parse mocap0/data.csv → sorted numpy arrays of timestamps and poses.

    Returns
    -------
    timestamps : (N,) float64  nanoseconds
    positions  : (N, 3) float64  metres
    rotmats    : (N, 3, 3) float64
    """
    ts, pos, quat = [], [], []
    with open(mocap_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            t_ns = int(parts[0])
            px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            ts.append(t_ns)
            pos.append([px, py, pz])
            quat.append([qx, qy, qz, qw])   # scipy xyzw format

    ts = np.array(ts, dtype=np.float64)
    pos = np.array(pos, dtype=np.float64)
    quat = np.array(quat, dtype=np.float64)

    # Sort by timestamp
    order = np.argsort(ts)
    ts, pos, quat = ts[order], pos[order], quat[order]

    # Convert quaternions to rotation matrices
    rotmats = Rotation.from_quat(quat).as_matrix()   # (N, 3, 3)
    return ts, pos, rotmats


def _load_cam_csv(cam_csv: str):
    """
    Parse camN/data.csv → list of (timestamp_ns, filename).
    """
    pairs = []
    with open(cam_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            pairs.append((int(parts[0]), parts[1].strip()))
    return pairs


def _interpolate_mocap(ts_query_ns, ts_mocap, pos_mocap, rot_mocap):
    """
    Linearly interpolate position and SLERP rotation at ts_query_ns.
    Returns (position, rotmat) or (None, None) if out of range.
    """
    if ts_query_ns < ts_mocap[0] or ts_query_ns > ts_mocap[-1]:
        return None, None

    idx = np.searchsorted(ts_mocap, ts_query_ns)
    idx = np.clip(idx, 1, len(ts_mocap) - 1)
    t0, t1 = ts_mocap[idx - 1], ts_mocap[idx]

    alpha = (ts_query_ns - t0) / (t1 - t0)
    pos = (1 - alpha) * pos_mocap[idx - 1] + alpha * pos_mocap[idx]

    # SLERP for rotation
    key_rots = Rotation.from_matrix(rot_mocap[[idx - 1, idx]])
    slerp = Slerp([0.0, 1.0], key_rots)
    rot = slerp(alpha).as_matrix()
    return pos, rot


def _load_calibration(calib_yaml: str):
    """
    Parse the TUM-RS calibration YAML.

    Returns
    -------
    K       : (3, 3) float64  pinhole intrinsics (undistorted)
    dist    : (4,) float64    equidistant distortion coefficients k1..k4
    T_cam_imu : (4, 4) float64  cam0→IMU extrinsic
    width, height : int
    """
    with open(calib_yaml) as f:
        data = yaml.safe_load(f)

    cam0 = data['cam0']
    fx, fy, cx, cy = cam0['intrinsics']
    k1, k2, k3, k4 = cam0['distortion_coeffs']
    width, height = cam0['resolution']

    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, k2, k3, k4], dtype=np.float64)

    # T_cam_imu is stored as a list of 4 rows
    T_cam_imu_list = cam0['T_cam_imu']
    T_cam_imu = np.array(T_cam_imu_list, dtype=np.float64)  # (4, 4)

    return K, dist, T_cam_imu, width, height


def _undistort_image_and_K(img_gray, K, dist, width, height):
    """
    Undistort a fisheye (equidistant) image.  Returns (undistorted, K_new).
    K_new is the new camera matrix for the rectified image.
    """
    # Build new intrinsics that avoids black borders
    K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, dist, (width, height), np.eye(3),
        balance=0.0, new_size=(width, height))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, dist, np.eye(3), K_new, (width, height), cv2.CV_16SC2)
    undist = cv2.remap(img_gray, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undist, K_new


class TumRS(_TorchDataset):
    """
    TUM Rolling-Shutter VIO dataset for relative pose evaluation.

    Parameters
    ----------
    root_dir : str
        Path to the root of the tum_rs dataset (contains euroc/, poses/,
        calibration/).
    sequences : list[int] or None
        Sequence indices 1–10 to include (default: all 10).
    strides : list[int]
        Frame-gap(s) for selecting pairs (default: [1, 5]).
    max_pairs : int
        Maximum pairs per (sequence, stride) combination (default: 500).
    """

    def __init__(self, root_dir, sequences=None, strides=None, max_pairs=500):
        self.root_dir = root_dir
        self.name = "TumRS"

        if sequences is None:
            sequences = list(range(1, 11))
        if strides is None:
            strides = [1, 5]

        calib_yaml = os.path.join(
            root_dir, 'calibration',
            'camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml')
        self.K_dist, self.dist, self.T_cam_imu, self.width, self.height = \
            _load_calibration(calib_yaml)

        # Compute K for undistorted images
        self.K_undist = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K_dist, self.dist,
            (self.width, self.height), np.eye(3),
            balance=0.0, new_size=(self.width, self.height))
        # Pre-build undistortion maps (shared across all images)
        self._map1, self._map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K_dist, self.dist, np.eye(3), self.K_undist,
            (self.width, self.height), cv2.CV_16SC2)

        self.data = []
        for seq in sequences:
            seq_dir = os.path.join(root_dir, 'euroc', f'dataset-seq{seq}')
            cam0_csv = os.path.join(seq_dir, 'mav0', 'cam0', 'data.csv')
            mocap_csv = os.path.join(seq_dir, 'mav0', 'mocap0', 'data.csv')
            cam0_imgdir = os.path.join(seq_dir, 'mav0', 'cam0', 'data')

            if not os.path.isfile(cam0_csv) or not os.path.isfile(mocap_csv):
                print(f"[TumRS] seq{seq}: missing data, skipping.")
                continue

            cam_frames = _load_cam_csv(cam0_csv)       # [(ts_ns, fname), ...]
            ts_mocap, pos_mocap, rot_mocap = _load_mocap(mocap_csv)
            T_imu_cam = np.linalg.inv(self.T_cam_imu)  # cam→world (via IMU)

            for stride in strides:
                pairs_added = 0
                step = max(1, len(cam_frames) // max_pairs) if len(cam_frames) > max_pairs * stride else 1

                for i in range(0, len(cam_frames) - stride, step):
                    if pairs_added >= max_pairs:
                        break
                    j = i + stride
                    ts1_ns, fname1 = cam_frames[i]
                    ts2_ns, fname2 = cam_frames[j]

                    # Get world poses for both frames
                    pos1, R_world_body1 = _interpolate_mocap(
                        ts1_ns, ts_mocap, pos_mocap, rot_mocap)
                    pos2, R_world_body2 = _interpolate_mocap(
                        ts2_ns, ts_mocap, pos_mocap, rot_mocap)
                    if pos1 is None or pos2 is None:
                        continue

                    # Transform from IMU/body frame to camera frame
                    # T_cam_imu: cam←imu (4×4)
                    R_cam_imu = self.T_cam_imu[:3, :3]
                    t_cam_imu = self.T_cam_imu[:3, 3]

                    # World→camera transform for each frame
                    R_w2c1 = R_cam_imu @ R_world_body1.T
                    t_w2c1 = R_cam_imu @ (-R_world_body1.T @ pos1) + t_cam_imu
                    R_w2c2 = R_cam_imu @ R_world_body2.T
                    t_w2c2 = R_cam_imu @ (-R_world_body2.T @ pos2) + t_cam_imu

                    # Relative pose: cam1→cam2
                    R_1_2 = R_w2c2 @ R_w2c1.T
                    t_1_2 = t_w2c2 - R_1_2 @ t_w2c1
                    # Normalise translation to unit vector (no metric scale)
                    t_norm = np.linalg.norm(t_1_2)
                    if t_norm < 1e-8:
                        continue
                    t_1_2_unit = t_1_2 / t_norm

                    # GT rolling-shutter velocities for frame 1 (camera frame)
                    # omega/v describe the camera motion *during* one full frame
                    # (row top → row bottom), parameterised by tau ∈ [-0.5, 0.5].
                    half_T = TUM_RS_READOUT_TIME_S * 0.5
                    ts1_start = ts1_ns - half_T * 1e9
                    ts1_end   = ts1_ns + half_T * 1e9
                    pos1s, R1s = _interpolate_mocap(
                        ts1_start, ts_mocap, pos_mocap, rot_mocap)
                    pos1e, R1e = _interpolate_mocap(
                        ts1_end,   ts_mocap, pos_mocap, rot_mocap)

                    ts2_start = ts2_ns - half_T * 1e9
                    ts2_end   = ts2_ns + half_T * 1e9
                    pos2s, R2s = _interpolate_mocap(
                        ts2_start, ts_mocap, pos_mocap, rot_mocap)
                    pos2e, R2e = _interpolate_mocap(
                        ts2_end,   ts_mocap, pos_mocap, rot_mocap)

                    # Compute omega_gt (rotation vector in camera frame)
                    # omega = log_SO3(R_cam_start^T @ R_cam_end)
                    def rs_omega_v(Rs, Re, ps, pe):
                        """Compute omega (rad) and v (m) in camera frame."""
                        if Rs is None or Re is None:
                            return np.zeros(3), np.zeros(3)
                        # Body-frame rotation during readout
                        dR_body = Rs.T @ Re
                        dR_cam  = R_cam_imu @ dR_body @ R_cam_imu.T
                        omega = Rotation.from_matrix(dR_cam).as_rotvec()
                        # Translation in camera frame
                        dp_world = pe - ps
                        dp_cam = R_cam_imu @ (R_world_body1.T @ dp_world)
                        return omega, dp_cam

                    omega1_gt, v1_gt = rs_omega_v(R1s, R1e, pos1s, pos1e)
                    omega2_gt, v2_gt = rs_omega_v(R2s, R2e, pos2s, pos2e)

                    self.data.append({
                        'seq': seq,
                        'stride': stride,
                        'id1': f'seq{seq}_{fname1[:-4]}',
                        'id2': f'seq{seq}_{fname2[:-4]}',
                        'img_path1': os.path.join(cam0_imgdir, fname1),
                        'img_path2': os.path.join(cam0_imgdir, fname2),
                        'R_1_2': R_1_2.astype(np.float64),
                        'T_1_2': t_1_2_unit.astype(np.float64),
                        'omega1_gt': omega1_gt.astype(np.float64),
                        'v1_gt': v1_gt.astype(np.float64),
                        'omega2_gt': omega2_gt.astype(np.float64),
                        'v2_gt': v2_gt.astype(np.float64),
                    })
                    pairs_added += 1

        print(f"[TumRS] Initialized with {len(self.data)} image pairs "
              f"(seqs={sequences}, strides={strides}).")

    def _load_and_undistort(self, path):
        """Load a grayscale image and apply fisheye undistortion."""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        undist = cv2.remap(img, self._map1, self._map2,
                           interpolation=cv2.INTER_LINEAR)
        return undist

    def get_dataloader(self):
        if not _HAS_TORCH:
            raise RuntimeError("get_dataloader() requires torch; use indexing instead")
        return DataLoader(
            self, batch_size=None, shuffle=False, pin_memory=False,
            num_workers=0, collate_fn=simple_collate_fn)

    def __getitem__(self, idx):
        d = self.data[idx]

        img1 = self._load_and_undistort(d['img_path1'])
        img2 = self._load_and_undistort(d['img_path2'])

        # Convert to RGB float tensor (C×H×W, [0,1])
        if _HAS_TORCH:
            img1_t = torch.from_numpy(
                np.stack([img1, img1, img1], axis=0)).float() / 255.0
            img2_t = torch.from_numpy(
                np.stack([img2, img2, img2], axis=0)).float() / 255.0
        else:
            img1_t = np.stack([img1, img1, img1], axis=0).astype(np.float32) / 255.0
            img2_t = np.stack([img2, img2, img2], axis=0).astype(np.float32) / 255.0

        return {
            'id1': d['id1'],
            'id2': d['id2'],
            'img1': img1_t,
            'img2': img2_t,
            'img1_np': img1,      # raw H×W uint8 for AC extractor
            'img2_np': img2,
            'K1': self.K_undist.copy(),
            'K2': self.K_undist.copy(),
            'R_1_2': d['R_1_2'],
            'T_1_2': d['T_1_2'],
            'omega1_gt': d['omega1_gt'],
            'v1_gt': d['v1_gt'],
            'omega2_gt': d['omega2_gt'],
            'v2_gt': d['v2_gt'],
            'seq': d['seq'],
            'stride': d['stride'],
            'height': self.height,
            'width': self.width,
        }

    def __len__(self):
        return len(self.data)

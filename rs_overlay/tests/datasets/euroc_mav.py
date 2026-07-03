"""
EuRoC MAV Dataset loader for relative pose evaluation.

EuRoC cameras are global-shutter. We compute "hypothetical" RS motion
parameters from the high-rate mocap trajectory so that RS solvers can
be evaluated:
  - R_1_2, T_1_2 : relative pose at the frame timestamps (GS ground truth)
  - omega_gt, v_gt : camera rotation/translation during one simulated RS
                     readout (≈ 30 ms), analogous to TUM-RS.

Because the images contain no actual RS distortion, RS solvers should
estimate omega ≈ 0, v ≈ 0, while R/t evaluation tests generalisation
of the method beyond TUM-RS.

Dataset layout (ASL format):
    root_dir/
        MH_01_easy/ (or V1_01_easy/, etc.)
            mav0/
                cam0/data/*.png      (left camera, 752×480, 20 fps)
                cam0/data.csv        (timestamp → filename)
                cam0/sensor.yaml     (intrinsics + distortion)
                state_groundtruth_estimate0/data.csv  (ground-truth poses)

Reference: Burri et al., "The EuRoC Micro Aerial Vehicle Datasets",
           IJRR 2016.
"""

import os
import csv
import cv2
import yaml
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

try:
    from torch.utils.data import Dataset as _TorchDataset
    import torch
    _HAS_TORCH = True
except ImportError:
    _TorchDataset = object
    torch = None
    _HAS_TORCH = False


# Simulated RS readout time (same as TUM-RS: ~30 ms)
SIMULATED_RS_READOUT_S = 30.0e-3


def _load_sensor_yaml(yaml_path):
    """Parse cam0/sensor.yaml → intrinsics, distortion, image size."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    fu, fv, cu, cv = data['intrinsics']
    K = np.array([[fu, 0, cu],
                  [0, fv, cv],
                  [0,  0,  1]], dtype=np.float64)
    dist_model = data.get('distortion_model', 'radial-tangential')
    dist_coeffs = np.array(data.get('distortion_coefficients', [0, 0, 0, 0]),
                           dtype=np.float64)
    w, h = data['resolution']
    # T_BS: cam0 → body (sensor extrinsic)
    T_BS = np.array(data['T_BS']['data'], dtype=np.float64).reshape(4, 4)
    return K, dist_coeffs, dist_model, w, h, T_BS


def _load_gt_poses(gt_csv):
    """
    Parse state_groundtruth_estimate0/data.csv.
    Format: timestamp [ns], p_x, p_y, p_z, q_w, q_x, q_y, q_z, v_x, v_y, v_z, ...

    Returns timestamps (ns), positions (N,3), rotmats (N,3,3).
    """
    ts, pos, quats = [], [], []
    with open(gt_csv) as f:
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
            quats.append([qx, qy, qz, qw])  # scipy xyzw

    ts = np.array(ts, dtype=np.float64)
    pos = np.array(pos, dtype=np.float64)
    quats = np.array(quats, dtype=np.float64)

    order = np.argsort(ts)
    ts, pos, quats = ts[order], pos[order], quats[order]
    rotmats = Rotation.from_quat(quats).as_matrix()
    return ts, pos, rotmats


def _load_cam_csv(cam_csv):
    """Parse cam0/data.csv → [(timestamp_ns, filename), ...]."""
    pairs = []
    with open(cam_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            pairs.append((int(parts[0]), parts[1].strip()))
    return pairs


def _interpolate_pose(ts_query_ns, ts_gt, pos_gt, rot_gt):
    """Interpolate pose at ts_query_ns via linear position + SLERP rotation."""
    if ts_query_ns < ts_gt[0] or ts_query_ns > ts_gt[-1]:
        return None, None
    idx = np.searchsorted(ts_gt, ts_query_ns)
    idx = np.clip(idx, 1, len(ts_gt) - 1)
    t0, t1 = ts_gt[idx - 1], ts_gt[idx]
    alpha = (ts_query_ns - t0) / (t1 - t0 + 1e-15)
    pos = (1 - alpha) * pos_gt[idx - 1] + alpha * pos_gt[idx]
    key_rots = Rotation.from_matrix(rot_gt[[idx - 1, idx]])
    slerp = Slerp([0.0, 1.0], key_rots)
    rot = slerp(np.clip(alpha, 0, 1)).as_matrix()
    return pos, rot


# Sequence names in the standard EuRoC distribution
EUROC_SEQUENCES = [
    'MH_01_easy', 'MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult',
    'V1_01_easy', 'V1_02_medium', 'V1_03_difficult',
    'V2_01_easy', 'V2_02_medium', 'V2_03_difficult',
]


class EuRoCMAV(_TorchDataset):
    """
    EuRoC MAV dataset for relative pose evaluation.

    Parameters
    ----------
    root_dir : str
        Path to the EuRoC root (contains MH_01_easy/, V1_01_easy/, etc.).
    sequences : list[str] or None
        Sequence names to include (default: all available on disk).
    strides : list[int]
        Frame-gap(s) for pair selection (default: [1, 5]).
    max_pairs : int
        Maximum pairs per (sequence, stride) combination.
    """

    def __init__(self, root_dir, sequences=None, strides=None, max_pairs=200):
        self.root_dir = root_dir
        self.name = "EuRoCMAV"

        if strides is None:
            strides = [10, 20]  # 500ms / 1000ms at 20fps → reasonable baselines

        # Auto-detect sequences on disk
        if sequences is None:
            sequences = [s for s in EUROC_SEQUENCES
                         if os.path.isdir(os.path.join(root_dir, s))]
            if not sequences:
                # Try nested structure (e.g., root_dir/mav0/...)
                raise FileNotFoundError(
                    f"No EuRoC sequences found in {root_dir}. "
                    f"Expected subdirectories like MH_01_easy/, V1_01_easy/, ...")

        self.data = []
        self._undist_maps = {}  # per-sequence undistortion maps
        self._K_undist = {}     # per-sequence undistorted K

        for seq_name in sequences:
            seq_dir = os.path.join(root_dir, seq_name)
            cam0_dir = os.path.join(seq_dir, 'mav0', 'cam0')
            sensor_yaml = os.path.join(cam0_dir, 'sensor.yaml')
            cam_csv = os.path.join(cam0_dir, 'data.csv')
            cam_imgdir = os.path.join(cam0_dir, 'data')

            # Ground truth: prefer state_groundtruth_estimate0, fallback to mocap0
            gt_csv = os.path.join(seq_dir, 'mav0',
                                  'state_groundtruth_estimate0', 'data.csv')
            if not os.path.isfile(gt_csv):
                gt_csv = os.path.join(seq_dir, 'mav0', 'mocap0', 'data.csv')

            if not all(os.path.isfile(f) for f in [sensor_yaml, cam_csv, gt_csv]):
                print(f"[EuRoCMAV] {seq_name}: missing files, skipping.")
                continue

            K, dist_coeffs, dist_model, width, height, T_BS = \
                _load_sensor_yaml(sensor_yaml)

            # Build undistortion maps
            if dist_model in ('radial-tangential', 'plumb_bob'):
                K_new, _ = cv2.getOptimalNewCameraMatrix(
                    K, dist_coeffs, (width, height), 0)
                map1, map2 = cv2.initUndistortRectifyMap(
                    K, dist_coeffs, None, K_new, (width, height), cv2.CV_16SC2)
            elif dist_model == 'equidistant':
                K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                    K, dist_coeffs, (width, height), np.eye(3),
                    balance=0.0, new_size=(width, height))
                map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                    K, dist_coeffs, np.eye(3), K_new,
                    (width, height), cv2.CV_16SC2)
            else:
                # No distortion
                K_new = K.copy()
                map1, map2 = None, None

            self._undist_maps[seq_name] = (map1, map2)
            self._K_undist[seq_name] = K_new
            self._dims = (width, height)

            # T_BS in EuRoC = sensor→body transform.
            # We need body→sensor (camera): T_cam_body = T_BS^{-1}
            T_cam_body = np.linalg.inv(T_BS)
            R_cam_body = T_cam_body[:3, :3]
            t_cam_body = T_cam_body[:3, 3]

            cam_frames = _load_cam_csv(cam_csv)
            ts_gt, pos_gt, rot_gt = _load_gt_poses(gt_csv)

            for stride in strides:
                pairs_added = 0
                step = max(1, len(cam_frames) // max_pairs) \
                    if len(cam_frames) > max_pairs * stride else 1

                for i in range(0, len(cam_frames) - stride, step):
                    if pairs_added >= max_pairs:
                        break
                    j = i + stride
                    ts1_ns, fname1 = cam_frames[i]
                    ts2_ns, fname2 = cam_frames[j]

                    # Interpolate world (body) poses
                    pos1, R_wb1 = _interpolate_pose(ts1_ns, ts_gt, pos_gt, rot_gt)
                    pos2, R_wb2 = _interpolate_pose(ts2_ns, ts_gt, pos_gt, rot_gt)
                    if pos1 is None or pos2 is None:
                        continue

                    # World→camera transforms
                    R_w2c1 = R_cam_body @ R_wb1.T
                    t_w2c1 = R_cam_body @ (-R_wb1.T @ pos1) + t_cam_body
                    R_w2c2 = R_cam_body @ R_wb2.T
                    t_w2c2 = R_cam_body @ (-R_wb2.T @ pos2) + t_cam_body

                    # Relative pose: cam1→cam2
                    R_1_2 = R_w2c2 @ R_w2c1.T
                    t_1_2 = t_w2c2 - R_1_2 @ t_w2c1
                    t_norm = np.linalg.norm(t_1_2)
                    if t_norm < 1e-8:
                        continue
                    t_1_2_unit = t_1_2 / t_norm

                    # Simulated RS parameters (what omega/v would be if
                    # the camera had rolling-shutter readout)
                    half_T = SIMULATED_RS_READOUT_S * 0.5

                    def _rs_omega_v(ts_ns, R_wb_ref):
                        ts_start = ts_ns - half_T * 1e9
                        ts_end = ts_ns + half_T * 1e9
                        pos_s, R_s = _interpolate_pose(ts_start, ts_gt, pos_gt, rot_gt)
                        pos_e, R_e = _interpolate_pose(ts_end, ts_gt, pos_gt, rot_gt)
                        if pos_s is None or pos_e is None:
                            return np.zeros(3), np.zeros(3)
                        dR_body = R_e.T @ R_s
                        dR_cam = R_cam_body @ dR_body @ R_cam_body.T
                        omega = Rotation.from_matrix(dR_cam).as_rotvec()
                        dp_world = pos_e - pos_s
                        dp_cam = R_cam_body @ (R_wb_ref.T @ dp_world)
                        return omega, dp_cam

                    omega1_gt, v1_gt = _rs_omega_v(ts1_ns, R_wb1)
                    omega2_gt, v2_gt = _rs_omega_v(ts2_ns, R_wb2)

                    self.data.append({
                        'seq_name': seq_name,
                        'seq': EUROC_SEQUENCES.index(seq_name) + 1
                               if seq_name in EUROC_SEQUENCES else 0,
                        'stride': stride,
                        'id1': f'{seq_name}_{fname1[:-4]}',
                        'id2': f'{seq_name}_{fname2[:-4]}',
                        'img_path1': os.path.join(cam_imgdir, fname1),
                        'img_path2': os.path.join(cam_imgdir, fname2),
                        'R_1_2': R_1_2.astype(np.float64),
                        'T_1_2': t_1_2_unit.astype(np.float64),
                        'omega1_gt': omega1_gt.astype(np.float64),
                        'v1_gt': v1_gt.astype(np.float64),
                        'omega2_gt': omega2_gt.astype(np.float64),
                        'v2_gt': v2_gt.astype(np.float64),
                    })
                    pairs_added += 1

        print(f"[EuRoCMAV] Initialized with {len(self.data)} image pairs "
              f"(seqs={[s for s in sequences if any(d['seq_name'] == s for d in self.data)]}, "
              f"strides={strides}).")

    def _load_and_undistort(self, path, seq_name):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        map1, map2 = self._undist_maps[seq_name]
        if map1 is not None:
            img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx):
        d = self.data[idx]
        seq_name = d['seq_name']

        img1 = self._load_and_undistort(d['img_path1'], seq_name)
        img2 = self._load_and_undistort(d['img_path2'], seq_name)
        K = self._K_undist[seq_name].copy()

        if _HAS_TORCH:
            img1_t = torch.from_numpy(
                np.stack([img1, img1, img1], axis=0)).float() / 255.0
            img2_t = torch.from_numpy(
                np.stack([img2, img2, img2], axis=0)).float() / 255.0
        else:
            img1_t = np.stack([img1, img1, img1], axis=0).astype(np.float32) / 255.0
            img2_t = np.stack([img2, img2, img2], axis=0).astype(np.float32) / 255.0

        H, W = img1.shape[:2]
        return {
            'id1': d['id1'],
            'id2': d['id2'],
            'img1': img1_t,
            'img2': img2_t,
            'img1_np': img1,
            'img2_np': img2,
            'K1': K,
            'K2': K,
            'R_1_2': d['R_1_2'],
            'T_1_2': d['T_1_2'],
            'omega1_gt': d['omega1_gt'],
            'v1_gt': d['v1_gt'],
            'omega2_gt': d['omega2_gt'],
            'v2_gt': d['v2_gt'],
            'seq': d['seq'],
            'stride': d['stride'],
            'height': H,
            'width': W,
        }

    def __len__(self):
        return len(self.data)

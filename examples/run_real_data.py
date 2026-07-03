#!/usr/bin/env python3
"""
Example (i): Real-data comparison on the TUM Rolling-Shutter dataset.

Runs our 7-AC rolling-shutter solver against the baselines
  * gs_5pt   — global-shutter 5-point (Nister)         [GS]
  * dai_20pt — Dai et al. CVPR'16 linear RS, 20 points [RS-20PC]
  * dai_44pt — Dai et al. CVPR'16 + GN, 44 points      [RS-44PC]
and prints an AUC@5/10/20 pose-accuracy table plus median RS-parameter
(angular/translational velocity) errors — reproducing the paper's real-data
comparison.

Affine correspondences are extracted from the RS image pairs with RoMa
(dense matcher). This step needs a CUDA GPU and the `romatch`, `torch`,
`kornia` packages (see INSTALL.md). Extracted ACs are cached to HDF5, so a
re-run reuses them without a GPU.

This is a thin wrapper around the validated tester at
  external/superansac/tests/essential_matrix/tester_rs_superansac.py
(run after the RS overlay has been applied). Pass --help to see all options;
any unknown option is forwarded to the tester.

Usage:
  # 1. get one sequence (~GBs):
  bash examples/download_tum_rs.sh datasets/tum_rs 1
  # 2. run the comparison:
  python examples/run_real_data.py --dataset_root datasets/tum_rs --sequences 1
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTER = os.path.join(
    REPO, "external", "superansac", "tests", "essential_matrix",
    "tester_rs_superansac.py")


def main():
    ap = argparse.ArgumentParser(
        description="Real-data RS relative-pose comparison (TUM-RS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset_root", default="datasets/tum_rs",
                    help="TUM-RS root (see examples/download_tum_rs.sh)")
    ap.add_argument("--sequences", type=int, nargs="+", default=[1])
    ap.add_argument("--strides", type=int, nargs="+", default=[10, 20],
                    help="frame strides (paper uses 10 and 20)")
    ap.add_argument("--max_pairs", type=int, default=100)
    ap.add_argument("--features", default="RoMa", choices=["RoMa", "KeyNetAffNetHardNet"],
                    help="AC extractor; paper headline uses RoMa")
    ap.add_argument("--methods", nargs="+",
                    default=["ac_rs_7direct", "gs_5pt", "dai_20pt", "dai_44pt"],
                    help="ac_rs_7direct = ours; the rest are the baselines")
    ap.add_argument("--scoring", default="MSAC")
    ap.add_argument("--sampler", default="PROSAC")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="inlier threshold in pixels")
    ap.add_argument("--max_iterations", type=int, default=1000)
    ap.add_argument("--output", default="out/results_rs_tum.csv")
    ap.add_argument("--device", default="cuda")
    args, extra = ap.parse_known_args()

    if not os.path.isfile(TESTER):
        sys.exit(f"Tester not found at {TESTER}\n"
                 "Run:  git submodule update --init external/superansac && "
                 "bash rs_overlay/apply_overlay.sh")
    if not os.path.isdir(os.path.join(args.dataset_root, "euroc")):
        sys.exit(f"No dataset under {args.dataset_root}/euroc.\n"
                 f"Run:  bash examples/download_tum_rs.sh {args.dataset_root} "
                 f"{' '.join(map(str, args.sequences))}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    cmd = [
        sys.executable, TESTER,
        "--dataset", "tum_rs",
        "--dataset_root", args.dataset_root,
        "--sequences", *map(str, args.sequences),
        "--strides", *map(str, args.strides),
        "--max_pairs", str(args.max_pairs),
        "--features", args.features,
        "--methods", *args.methods,
        "--scoring", args.scoring,
        "--sampler", args.sampler,
        "--threshold", str(args.threshold),
        "--max_iterations", str(args.max_iterations),
        "--output", args.output,
        "--device", args.device,
        *extra,
    ]
    print("Running:", " ".join(cmd), flush=True)
    # Run from the tester's directory so its relative imports / module paths resolve.
    raise SystemExit(subprocess.call(cmd, cwd=os.path.dirname(TESTER)))


if __name__ == "__main__":
    main()

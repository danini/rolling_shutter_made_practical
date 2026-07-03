#!/usr/bin/env bash
# =====================================================================
# One-shot setup for Rolling Shutter Relative Pose Estimation Made Practical.
#
#   1. initialize the pinned SupeRANSAC submodule
#   2. apply the RS overlay onto it
#   3. build the C++ synthetic benchmark (CMake)
#   4. (optional) install the Python module for the real-data example
#
# Usage:
#   bash setup.sh                 # steps 1-3 (C++ benchmark only)
#   bash setup.sh --with-python   # steps 1-4 (also `pip install` pysuperansac)
#   bash setup.sh --native        # add -march=native -ffast-math (paper timings)
#
# See INSTALL.md for system prerequisites (OpenCV, Eigen, Boost, ...).
# =====================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

WITH_PYTHON=0
CMAKE_FLAGS=()
for arg in "$@"; do
  case "$arg" in
    --with-python) WITH_PYTHON=1 ;;
    --native)      CMAKE_FLAGS+=(-DRS_ENABLE_NATIVE_ARCH=ON -DRS_ENABLE_FAST_MATH=ON) ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

echo "==> [1/4] Initializing SupeRANSAC submodule"
git submodule update --init --recursive external/superansac

echo "==> [2/4] Applying RS overlay"
bash rs_overlay/apply_overlay.sh

echo "==> [3/4] Building C++ synthetic benchmark"
cmake -B build -DCMAKE_BUILD_TYPE=Release "${CMAKE_FLAGS[@]}"
cmake --build build -j"$(nproc)"
echo "    -> build/benchmark_rs_superansac"

if [[ "${WITH_PYTHON}" -eq 1 ]]; then
  echo "==> [4/4] Installing Python module (pysuperansac)"
  pip install ./external/superansac
  echo "    -> import pysuperansac"
else
  echo "==> [4/4] Skipped Python module (pass --with-python to enable)"
fi

echo "Setup complete."

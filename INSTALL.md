# Installation Guide

This repo has two independent build targets:

1. **C++ synthetic benchmark** (`build/benchmark_rs_superansac`) — for the
   synthetic example. Needs only a C++ toolchain + OpenCV/Eigen/Boost.
2. **Python module** (`pysuperansac`) — for the real-data example. Additionally
   needs pybind11 + Python, and (for AC extraction) a CUDA GPU with
   `torch`/`kornia`/`romatch`.

Both are driven by `setup.sh`. You can build only what you need.

---

## 1. System prerequisites (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential cmake git \
    libeigen3-dev libboost-all-dev \
    libopencv-dev libopencv-contrib-dev
```

- **CMake** ≥ 3.16, a **C++17** compiler (GCC ≥ 9 or Clang ≥ 10).
- **OpenCV** ≥ 4.x (with `contrib`), **Eigen3**, **Boost** (the `system`
  component). No BLAS/LAPACK or TBB required.

On other platforms install the equivalents (e.g. `brew install opencv eigen boost`
on macOS). If OpenCV/Boost are in a non-standard prefix, pass
`-DOpenCV_DIR=...` / `-DBOOST_ROOT=...` to CMake.

---

## 2. Clone with the submodule

```bash
git clone --recursive https://github.com/danini/rolling_shutter_made_practical.git
cd rolling_shutter_made_practical
```

If you cloned without `--recursive`:

```bash
git submodule update --init --recursive external/superansac
```

The submodule (`external/superansac`) is pinned to the exact SupeRANSAC commit
this code was validated against.

---

## 3. The RS overlay (how the solver is delivered)

The rolling-shutter solver lives inside SupeRANSAC's estimator framework. Rather
than forking SupeRANSAC, we keep it a clean pinned submodule and ship the RS code
as an **overlay** under `rs_overlay/`:

- new RS estimator headers (`rs_overlay/include/estimators/*rs*`),
- the RS pybind bindings (`rs_overlay/python/pybind_rs_functions.cpp`, `pybind_helpers.h`),
- the TUM-RS / EuRoC loaders and real-data tester (`rs_overlay/tests/`),
- full-file copies of a handful of base SupeRANSAC files carrying small
  RS-integration edits.

`rs_overlay/apply_overlay.sh` copies these into `external/superansac/` before the
build. It is **idempotent** and intentionally leaves the submodule working tree
"dirty" (those changes are never committed back to the submodule). `setup.sh`
runs it for you.

---

## 4. Build the C++ benchmark

```bash
bash setup.sh
```

This runs: submodule init → `apply_overlay.sh` → CMake build. The result is
`build/benchmark_rs_superansac`. Equivalent manual steps:

```bash
git submodule update --init --recursive external/superansac
bash rs_overlay/apply_overlay.sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

### Portability vs. speed flags

For portability, `-march=native` and `-ffast-math` are **off by default**. To
reproduce the paper's timings on the build host:

```bash
bash setup.sh --native           # or: cmake -B build -DRS_ENABLE_NATIVE_ARCH=ON -DRS_ENABLE_FAST_MATH=ON
```

> ⚠️ `-ffast-math` changes IEEE/NaN semantics. The solver is written to be
> correct under it, but if you extend the code, be aware that `x != NaN` may not
> behave as expected. Leave it off unless you are benchmarking.

Run it:

```bash
./build/benchmark_rs_superansac --method ac_rs_7direct --trials 20
# methods: ac_rs_7direct (ours), gs_5pt, dai_20pt, dai_44pt, pc_rs, ...  or 'all'
```

---

## 5. Build the Python module

```bash
bash setup.sh --with-python      # or: pip install ./external/superansac
```

Requires `pybind11` and Python headers:

```bash
pip install pybind11 numpy
# system Python headers if needed: sudo apt-get install python3-dev
```

Verify:

```bash
python -c "import pysuperansac; print(hasattr(pysuperansac, 'estimateRSEssentialMatrix7AC'))"
# -> True
```

### GPU / feature-extraction dependencies (real data only)

Extracting affine correspondences from real images uses **RoMa** (dense
matcher) and needs a CUDA GPU:

```bash
pip install torch torchvision          # matching your CUDA version
pip install kornia kornia_moons h5py pandas matplotlib tqdm
pip install romatch                    # RoMa dense matcher
```

Extracted ACs are cached to HDF5, so **re-runs of the same sequence do not need
a GPU**. AffNet (`--features KeyNetAffNetHardNet`) is a lighter alternative that
uses `kornia` without RoMa.

---

## 6. Download real data

```bash
bash examples/download_tum_rs.sh datasets/tum_rs 1     # sequence 1 (~GBs)
```

This fetches the EuRoC/ASL-format export of the
[TUM Rolling-Shutter dataset](https://cvg.cit.tum.de/data/datasets/rolling-shutter-dataset)
and arranges it as the loader expects:

```
datasets/tum_rs/
  euroc/dataset-seq1/mav0/{cam0,cam1,mocap0}/...
  calibration/camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml
```

Then run the comparison:

```bash
python examples/run_real_data.py --dataset_root datasets/tum_rs --sequences 1
```

---

## Troubleshooting

- **`external/superansac is empty`** — run
  `git submodule update --init --recursive external/superansac`.
- **`RS overlay not applied` / missing `*rs*` headers** — run
  `bash rs_overlay/apply_overlay.sh`.
- **`fatal error: pybind_helpers.h`** — the overlay was not applied before
  `pip install`; run `apply_overlay.sh` first (or use `setup.sh --with-python`).
- **OpenCV/Boost not found** — pass `-DOpenCV_DIR=...` / `-DBOOST_ROOT=...`.

#!/usr/bin/env bash
# =====================================================================
# Download one (or more) TUM Rolling-Shutter sequence(s) into the layout
# expected by the loader (rs_overlay/tests/datasets/tum_rs.py).
#
# Source: TUM Rolling-Shutter Dataset (Schubert et al., IROS 2018),
#         https://cvg.cit.tum.de/data/datasets/rolling-shutter-dataset
# We use the EuRoC/ASL-format export (.tar), which the loader reads directly.
#
# Usage:
#   bash examples/download_tum_rs.sh [DEST_DIR] [SEQ ...]
#     DEST_DIR : where to place the dataset       (default: datasets/tum_rs)
#     SEQ      : sequence numbers to fetch 1..10  (default: 1)
#
# Resulting layout:
#   DEST_DIR/
#     euroc/dataset-seq{N}/mav0/{cam0,cam1,mocap0}/...
#     calibration/camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml
#
# Note: each sequence is several GB. Sequence 1 is enough for the demo.
# =====================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

DEST="${1:-datasets/tum_rs}"
shift || true
SEQS=("$@")
if [[ ${#SEQS[@]} -eq 0 ]]; then SEQS=(1); fi

BASE_EUROC="https://cdn3.vision.in.tum.de/rolling/exported/euroc"
CALIB_URL="https://cdn3.vision.in.tum.de/rolling/calibration/camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml"

DL() {  # DL <url> <output>
  if command -v wget >/dev/null 2>&1; then
    wget -c -O "$2" "$1"
  else
    curl -L -C - -o "$2" "$1"
  fi
}

mkdir -p "${DEST}/euroc" "${DEST}/calibration"

echo "==> Calibration YAML"
DL "${CALIB_URL}" "${DEST}/calibration/camchain-calibration-equidistant4_camimu_dataset-calib-imu1.yaml"

for s in "${SEQS[@]}"; do
  tar_url="${BASE_EUROC}/dataset-seq${s}.tar"
  tar_path="${DEST}/dataset-seq${s}.tar"
  seq_dir="${DEST}/euroc/dataset-seq${s}"
  if [[ -d "${seq_dir}/mav0/cam0" ]]; then
    echo "==> seq${s} already present, skipping download"
    continue
  fi
  echo "==> Downloading seq${s}  (${tar_url})"
  DL "${tar_url}" "${tar_path}"
  # Verify the MD5 checksum published alongside the tar.
  if DL "${tar_url}.md5" "${tar_path}.md5" 2>/dev/null; then
    ( cd "${DEST}" && md5sum -c "dataset-seq${s}.tar.md5" ) \
      || { echo "MD5 check FAILED for seq${s}" >&2; exit 1; }
    rm -f "${tar_path}.md5"
  fi
  echo "==> Extracting seq${s} into ${DEST}/euroc/"
  # The EuRoC export tar contains a top-level dataset-seq{N}/ directory
  # (with mav0/ and dso/ subfolders); the loader reads mav0/.
  tar -xf "${tar_path}" -C "${DEST}/euroc/"
  rm -f "${tar_path}"
  if [[ ! -d "${seq_dir}/mav0/cam0" ]]; then
    echo "WARNING: expected ${seq_dir}/mav0/cam0 after extraction — check the tar layout." >&2
  fi
done

echo "Done. Dataset root: ${DEST}"
echo "Pass it to the example with:  --dataset_root ${DEST}"

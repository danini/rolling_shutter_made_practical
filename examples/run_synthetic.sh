#!/usr/bin/env bash
# =====================================================================
# Example (ii): Synthetic experiments + plots.
#
# Runs the C++ benchmark on synthetic rolling-shutter scenes for our 7-AC
# solver and the baselines (GS 5-point, Dai RS-20PC, Dai RS-44PC), sweeping
# RS magnitude, point/affine noise, correspondence count and outlier ratio,
# then plots accuracy and runtime.
#
# Usage:
#   bash examples/run_synthetic.sh [TRIALS]      # default 50 trials
#
# Output: out/synthetic.csv and PNG/PDF figures in out/plots/.
# =====================================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

TRIALS="${1:-50}"
BIN=build/benchmark_rs_superansac
OUT=out
CSV="${OUT}/synthetic.csv"

if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: ${BIN} not found. Build first:  bash setup.sh" >&2
  exit 1
fi

mkdir -p "${OUT}"

# Methods to compare (paper baselines + ours). Each writes the same CSV
# schema; we keep the header from the first run and append the rest.
METHODS=(ac_rs_7direct gs_5pt dai_20pt dai_44pt)

echo "Running synthetic benchmark (${TRIALS} trials/method): ${METHODS[*]}"
: > "${CSV}"
first=1
for m in "${METHODS[@]}"; do
  echo "  -> ${m}"
  if [[ ${first} -eq 1 ]]; then
    "${BIN}" --method "${m}" --trials "${TRIALS}" >> "${CSV}"
    first=0
  else
    # drop the repeated header line for subsequent methods
    "${BIN}" --method "${m}" --trials "${TRIALS}" | tail -n +2 >> "${CSV}"
  fi
done

echo "Wrote ${CSV}"
echo "Plotting -> ${OUT}/plots/"
python3 examples/plot_benchmark.py "${CSV}" --outdir "${OUT}/plots"
echo "Done. See ${OUT}/plots/ (benchmark_pose, benchmark_rs, benchmark_runtime, ...)."

#!/usr/bin/env bash
# =====================================================================
# Apply the rolling-shutter overlay onto the pinned SupeRANSAC submodule.
#
# The RS solver is delivered as an overlay on top of the pristine
# danini/superansac submodule (so we do not have to push the RS code
# upstream). This script copies:
#   * 9 new RS estimator headers                (include/estimators/)
#   * the RS pybind bindings                    (python/pybind_rs_functions.cpp)
#   * the RS dataset loaders + testers          (tests/)
#   * full-file replacements of 8 upstream files that carry the small
#     RS-integration edits (headers, models, sampler, pybind glue)
# into external/superansac/. Copying (not patching) is intentional: the
# submodule is pinned to the exact commit these files were built against,
# so overlaying reproduces the validated build with no patch fuzz.
#
# Running it twice is harmless (idempotent overwrite). It leaves the
# submodule working tree "dirty" by design.
# =====================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
SA="${REPO_ROOT}/external/superansac"

if [[ ! -f "${SA}/src/superansac.cpp" ]]; then
  echo "ERROR: ${SA} looks empty." >&2
  echo "Run:  git submodule update --init external/superansac" >&2
  exit 1
fi

echo "Applying RS overlay -> ${SA}"

# Mirror the overlay tree (include/, python/, tests/) into the submodule.
# --no-perms keeps the submodule's own permissions; we only need contents.
copy_tree() {
  local sub="$1"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${HERE}/${sub}/" "${SA}/${sub}/"
  else
    cp -a "${HERE}/${sub}/." "${SA}/${sub}/"
  fi
}

copy_tree include
copy_tree python
copy_tree tests

echo "Overlay applied. Files now present in the submodule:"
ls "${SA}/include/estimators/" | grep -E "rs_|_rs_|seven_ac" || true
echo "Done."

#!/usr/bin/env sh
# ==============================================================================
# Copyright Contributors to the oneDAL Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Per-output packager for the scikit-learn-intelex conda split.
# The top-level build.sh already installed the full package into
# ${SKLEARNEX_STAGE}/<setuptools-sys.prefix>/... .  This script copies the
# subset that belongs to the current $PKG_NAME into $PREFIX.
#
# Split boundary:
#   - scikit-learn-intelex       : all Python sources, _onedal_py_host.so,
#                                  _daal4py.so, mpi_transceiver.so, metadata.
#   - scikit-learn-intelex-gpu   : _onedal_py_dpc*.so and _onedal_py_spmd_dpc*.so
#                                  (the backends that link against the SYCL runtime).
#
# Note: conda-build invokes the top-level build.sh (once) and each output's
# pack.sh with DIFFERENT $PREFIX placeholder values.  So we cannot assume the
# staged tree sits at ${SKLEARNEX_STAGE}${PREFIX}: we locate it by searching
# inside the staging root instead.

set -e

STAGE="${SKLEARNEX_STAGE:-${SRC_DIR}/__sklearnex_stage}"

if [ ! -d "${STAGE}" ]; then
    echo "pack.sh: staging root not found at ${STAGE}" >&2
    echo "pack.sh: expected top-level build.sh to have populated it via 'setup.py install --root ${STAGE}'" >&2
    exit 1
fi

# Locate the staged onedal package directory (only path we need to split at
# file granularity). This is the authoritative anchor for the staged layout.
STAGED_ONEDAL=$(find "${STAGE}" -type d -name onedal -path "*/site-packages/onedal" | head -1)
if [ -z "${STAGED_ONEDAL}" ]; then
    echo "pack.sh: staged onedal package not found under ${STAGE}" >&2
    exit 1
fi

# Derive the staged sys.prefix: .../<sys.prefix>/lib/pythonX.Y/site-packages/onedal
# -> strip the trailing "/lib/python*/site-packages/onedal".
STAGED_PREFIX=$(echo "${STAGED_ONEDAL}" | sed -E 's#/lib/python[0-9.]+/site-packages/onedal$##')

# Path under $PREFIX where the onedal package should land.
RELATIVE_ONEDAL="${STAGED_ONEDAL#${STAGED_PREFIX}/}"
TARGET_ONEDAL="${PREFIX}/${RELATIVE_ONEDAL}"

case "${PKG_NAME}" in
    scikit-learn-intelex)
        # Copy everything from the staged sys.prefix into $PREFIX, then strip
        # the DPC backend .so files — they belong to scikit-learn-intelex-gpu.
        cp -a "${STAGED_PREFIX}/." "${PREFIX}/"
        find "${TARGET_ONEDAL}" -maxdepth 1 -type f \
            \( -name '_onedal_py_dpc*' -o -name '_onedal_py_spmd_dpc*' \) \
            -exec rm -f {} +
        ;;

    scikit-learn-intelex-gpu)
        # Copy ONLY the DPC backend modules into onedal/.  The rest of the
        # package (Python sources, host backend, metadata) is already provided
        # by the scikit-learn-intelex run dependency (exact pin).
        mkdir -p "${TARGET_ONEDAL}"
        found_dpc=0
        found_spmd=0
        for so in "${STAGED_ONEDAL}"/_onedal_py_dpc*; do
            [ -e "${so}" ] || continue
            cp -P "${so}" "${TARGET_ONEDAL}/"
            found_dpc=1
        done
        for so in "${STAGED_ONEDAL}"/_onedal_py_spmd_dpc*; do
            [ -e "${so}" ] || continue
            cp -P "${so}" "${TARGET_ONEDAL}/"
            found_spmd=1
        done
        # SPMD backend is only built when NO_DIST is unset (setup.py gates
        # _onedal_py_spmd_dpc on build_distributed).
        if [ "${found_dpc}" = "0" ] || { [ -z "${NO_DIST}" ] && [ "${found_spmd}" = "0" ]; }; then
            echo "pack.sh: missing DPC backend .so files in ${STAGED_ONEDAL} (dpc=${found_dpc}, spmd=${found_spmd}, NO_DIST=${NO_DIST})" >&2
            echo "pack.sh: the top-level build did not produce a full DPC backend — check DPCPPROOT and compiler detection in setup.py" >&2
            exit 1
        fi
        ;;

    *)
        echo "pack.sh: unknown PKG_NAME='${PKG_NAME}'" >&2
        exit 1
        ;;
esac

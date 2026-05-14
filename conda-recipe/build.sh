#!/bin/bash
#===============================================================================
# Copyright 2018 Intel Corporation
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
#===============================================================================

# Top-level build.sh runs once per conda-build invocation and installs the full
# scikit-learn-intelex package (both _onedal_py_host and _onedal_py_dpc backends)
# into a staging prefix.  Per-output pack.sh scripts then copy the appropriate
# subset into $PREFIX.  This mirrors oneDAL's build.sh + pack.sh layout
# introduced in https://github.com/uxlfoundation/oneDAL/pull/3551 .

if [ -z "${PYTHON}" ]; then
    export PYTHON=python
fi

if [ ! -z "${PKG_VERSION}" ]; then
    export SKLEARNEX_VERSION=$PKG_VERSION
fi

if [ -z "${DALROOT}" ]; then
    export DALROOT=${PREFIX}
elif [ "${DALROOT}" != "${CONDA_PREFIX}" ] && [ ! -z "${CONDA_PREFIX}" ]; then
    # source oneDAL if DALROOT is set outside of conda-build
    source ${DALROOT}/env/vars.sh
    export DALRPATH=--abs-rpath
fi

if [ -z "${MPIROOT}" ] && [ -z "${NO_DIST}" ]; then
    export MPIROOT=${PREFIX}
fi
# reset preferred compilers to avoid usage of icx/icpx by default in all cases
if [ ! -z "${CC_FOR_BUILD}" ] && [ ! -z "${CXX_FOR_BUILD}" ]; then
    export CC=$CC_FOR_BUILD
    export CXX=$CXX_FOR_BUILD
fi
# source compiler if DPCPPROOT is set outside of conda-build
if [ ! -z "${DPCPPROOT}" ]; then
    source ${DPCPPROOT}/env/vars.sh
fi

# Install into a staging directory so each output's pack.sh can pick the
# subset of files it needs.  Use $SRC_DIR (the source work directory) rather
# than $BUILD_PREFIX: conda-build rebuilds $BUILD_PREFIX between the top-level
# build phase and each output phase, which would wipe the staging tree.
# $SRC_DIR is the current working directory for build.sh and pack.sh alike,
# and conda-build preserves it across output phases.  Mirrors oneDAL's
# pack.sh pattern (which reads __release_* out of $SRC_DIR/work).
export SKLEARNEX_STAGE="${SRC_DIR}/__sklearnex_stage"
rm -rf "${SKLEARNEX_STAGE}"
mkdir -p "${SKLEARNEX_STAGE}"

# Install under the staging root. Do NOT pass --prefix: conda-build runs
# build.sh (this top-level script) and each output's pack.sh with DIFFERENT
# $PREFIX placeholder values, so a hard-coded prefix here would not match
# what pack.sh sees. Letting setuptools use its own sys.prefix keeps the
# staged tree self-contained — pack.sh locates site-packages/onedal by
# searching the staging root.
${PYTHON} setup.py install \
    --single-version-externally-managed \
    --record "${SKLEARNEX_STAGE}/record.txt" \
    --root "${SKLEARNEX_STAGE}" \
    ${DALRPATH}

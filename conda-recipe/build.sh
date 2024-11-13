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

if [ -z "${PYTHON}" ]; then
    export PYTHON=python
fi

if [ ! -z "${PKG_VERSION}" ]; then
    export SKLEARNEX_VERSION=$PKG_VERSION
fi

if [ -z "${DALROOT}" ]; then
    export DALROOT=${PREFIX}
elif [ "${DALROOT}" != "${CONDA_PREFIX}" ]; then
    # source oneDAL if DALROOT is set outside of conda-build
    source ${DALROOT}/env/vars.sh
fi

if [ -n "${TBBROOT}" ] && [ "${TBBROOT}" != "${CONDA_PREFIX}" ]; then
# source TBB if TBBROOT is set outside of conda-build
    source ${TBBROOT}/env/vars.sh
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

${PYTHON} setup.py install --single-version-externally-managed --record record.txt

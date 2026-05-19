#!/bin/bash
#===============================================================================
# Copyright 2024 Intel Corporation
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

# if any parameter is given then only source TBB from the oneAPI install
if [ $# -eq 0 ]; then
  source /opt/intel/oneapi/setvars.sh
  # Pin dpctl to the Intel CPU OpenCL runtime so it doesn't fall back to a host ICD
  INTEL_OCL=$(find /opt/intel/oneapi -name libintelocl.so 2>/dev/null | head -1)
  if [ -n "$INTEL_OCL" ]; then export OCL_ICD_FILENAMES=$INTEL_OCL; fi
else
  source /opt/intel/oneapi/tbb/latest/env/vars.sh # prepare tbb
fi
source ./__release_lnx/daal/latest/env/vars.sh # prepare oneDAL

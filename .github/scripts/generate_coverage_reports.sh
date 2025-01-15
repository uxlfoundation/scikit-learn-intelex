#===============================================================================
# Copyright Contributors to the oneDAL project
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

ci_dir=$(dirname $(dirname $(dirname "${BASH_SOURCE[0]}")))
cd $ci_dir

# create coverage.py report
coverage combine .coverage.sklearnex .coverage.sklearn
coverage lcov -o coverage_py_"${1}".info

# create gcov report (lcov format)
if [[ -n "${SKLEARNEX_GCOV}" ]]; then
    # extract llvm tool for gcov processing
    if [[ -z "$2" ]]; then
        GCOV_EXE="$(dirname $(type -P -a icx))/compiler/llvm-cov gcov"
    else
        GCOV_EXE="gcov"
    fi
    echo $GCOV_EXE
    FILTER=$(realpath ./onedal).*
    echo $FILTER
    
    NUMPY_TEST=$(python -m pip freeze | grep numpy)
    # install dependencies
    # proper operation of gcov with sklearnex requires the header files from
    # the build numpy, this must be previously set as NUMPY_BUILD
    python -m pip install gcovr $NUMPY_BUILD
    
    gcovr --gcov-executable "${GCOV_EXE}" -r . -v --lcov --filter "${FILTER}" -o coverage_cpp_"${1}".info
    
    # reinstall previous numpy
    python -m pip install $NUMPY_TEST
fi

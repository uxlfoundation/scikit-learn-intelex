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

sklex_root="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
count=3
while [[ count -ne 0 ]]; do
    if [[ -d $sklex_root/.ci/ && -d $sklex_root/examples/ && -d $sklex_root/tests/ ]]; then
        break
    fi
    sklex_root="$( dirname "${sklex_root}" )"
    count=$(($count - 1))
done

if [[ count -eq 0 ]]; then
    echo "run_test.sh did not find the required testing directories"
    exit 1
fi

return_code=0

if [ -z "${PYTHON}" ]; then
    export PYTHON=python
fi

# Note: execute with argument --json-report in order to produce
# a JSON report under folder '.pytest_reports'. Other arguments
# will also be forwarded to pytest.
with_json_report=0
if [[ "$*" == *"--json-report"* ]]; then
    echo "Will produce JSON report of tests"
    with_json_report=1
    mkdir -p .pytest_reports
    if [[ ! -z "$(ls .pytest_reports)" ]]; then
        rm .pytest_reports/*.json
    fi
fi

function generate_pytest_args {
    declare -a ARGS=()
    if [[ "${with_json_report}" == "1" ]]; then
        ARGS+=("--json-report-file=.pytest_reports/$1_report.json")
    fi
    if [ -n "${COVERAGE_RCFILE}" ]; then
        ARGS+=(--cov=onedal --cov=sklearnex --cov-config="${COVERAGE_RCFILE}" --cov-append --cov-branch --cov-report=)
    fi
    printf -- "${ARGS[*]}"
}

${PYTHON} -c "from sklearnex import patch_sklearn; patch_sklearn()"
return_code=$(($return_code + $?))

pytest --verbose -s "${sklex_root}/tests" $@ $(generate_pytest_args legacy)
return_code=$(($return_code + $?))

pytest --verbose --pyargs daal4py $@ $(generate_pytest_args daal4py)
return_code=$(($return_code + $?))

pytest --verbose --pyargs sklearnex $@ $(generate_pytest_args sklearnex)
return_code=$(($return_code + $?))

pytest --verbose --pyargs onedal $@ $(generate_pytest_args onedal)
return_code=$(($return_code + $?))

pytest --verbose -s "${sklex_root}/.ci/scripts/test_global_patch.py" $@ $(generate_pytest_args global_patching)
return_code=$(($return_code + $?))

echo "NO_DIST=$NO_DIST"
if [[ ! $NO_DIST ]]; then
    mpirun --version
    # Note: OpenMPI will not allow running more processes than there
    # are cores in the machine, and Intel's MPI doesn't support the
    # same command line options, hence this line.
    if [[ ! -z "$(mpirun -h | grep "Open MPI")" ]]; then
        export EXTRA_MPI_ARGS="-n 2 -oversubscribe"
    else
        export EXTRA_MPI_ARGS="-n 4"
    fi
    mpirun ${EXTRA_MPI_ARGS} python "${sklex_root}/tests/helper_mpi_tests.py" \
        pytest -k spmd --with-mpi --verbose --pyargs sklearnex $@ $(generate_pytest_args sklearnex_spmd)
    return_code=$(($return_code + $?))
    mpirun ${EXTRA_MPI_ARGS} python "${sklex_root}/tests/helper_mpi_tests.py" \
        pytest --verbose -s "${sklex_root}/tests/test_daal4py_spmd_examples.py" $@ $(generate_pytest_args mpi_legacy)
    return_code=$(($return_code + $?))
fi

if [[ "$*" == *"--json-report"* ]] && ! [ -f .pytest_reports/legacy_report.json ]; then
    echo "Error: JSON report files failed to be produced."
    return_code=$(($return_code + 1))
fi

exit $return_code

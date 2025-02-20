@echo on
rem ============================================================================
rem Copyright 2018 Intel Corporation
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     http://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ============================================================================

rem %1 - scikit-learn-intelex repo root (should end with '\', leave empty if it's %cd% / $PWD)

set exitcode=0

setlocal enableextensions
IF NOT DEFINED PYTHON (
    set "PYTHON=python"
    set NO_DIST=1
)
if "%PYTHON%"=="python" (
    set NO_DIST=1
)



%PYTHON% -c "from sklearnex import patch_sklearn; patch_sklearn()" || set exitcode=1

set "PYTEST_ARGS= "

IF DEFINED COVERAGE_RCFILE (set "PYTEST_ARGS=--cov=onedal --cov=sklearnex --cov-config=%COVERAGE_RCFILE% --cov-append --cov-branch --cov-report= %PYTEST_ARGS%")

rem Note: execute with argument --json-report as second argument
rem in order to produce a JSON report under folder '.pytest_reports'.
if "%~2"=="--json-report" (
    set "PYTEST_ARGS=--json-report --json-report-file=.pytest_reports\FILENAME.json %PYTEST_ARGS%"
    echo %PYTEST_ARGS%
    mkdir .pytest_reports
    del /q .pytest_reports\*.json
)

echo "NO_DIST=%NO_DIST%"
setlocal enabledelayedexpansion
pytest --verbose -s "%1tests" %PYTEST_ARGS:FILENAME=legacy_report% || set exitcode=1
pytest --verbose --pyargs daal4py %PYTEST_ARGS:FILENAME=daal4py_report% || set exitcode=1
pytest --verbose --pyargs sklearnex %PYTEST_ARGS:FILENAME=sklearnex_report% || set exitcode=1
pytest --verbose --pyargs onedal %PYTEST_ARGS:FILENAME=onedal_report% || set exitcode=1
pytest --verbose "%1.ci\scripts\test_global_patch.py" %PYTEST_ARGS:FILENAME=global_patching_report% || set exitcode=1
if NOT "%NO_DIST%"=="1" (
    %PYTHON% "%1tests\helper_mpi_tests.py"^
        pytest -k spmd --with-mpi --verbose -s --pyargs sklearnex %PYTEST_ARGS:FILENAME=sklearnex_spmd%
    if !errorlevel! NEQ 0 (
        set exitcode=1
    )
    %PYTHON% "%1tests\helper_mpi_tests.py"^
        pytest --with-mpi --verbose -s "%1tests\test_daal4py_spmd_examples.py" %PYTEST_ARGS:FILENAME=mpi_legacy%
    if !errorlevel! NEQ 0 (
        set exitcode=1
    )
)
if "%~2"=="--json-report" (
    if NOT EXIST .pytest_reports\legacy_report.json (
        echo "Error: JSON report files failed to be produced."
        set exitcode=1
    )
)
EXIT /B %exitcode%

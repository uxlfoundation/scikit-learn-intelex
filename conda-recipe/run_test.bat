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

set SCIPY_ARRAY_API=1

%PYTHON% -c "from sklearnex import patch_sklearn; patch_sklearn()" || set exitcode=1

set "PYTEST_ARGS= "
set "PYTEST_CONFIG=-c %1setup.cfg"

IF DEFINED COVERAGE_RCFILE (set "PYTEST_ARGS=--cov=onedal --cov=sklearnex --cov-config=%COVERAGE_RCFILE% --cov-append --cov-branch --cov-report= %PYTEST_ARGS%")

rem Note: execute with argument --json-report as second argument
rem in order to produce a JSON report under folder '.pytest_reports'.
if "%~2"=="--json-report" (
    set "PYTEST_ARGS=--json-report --json-report-file=.pytest_reports\FILENAME.json %PYTEST_ARGS%"
    echo %PYTEST_ARGS%
    mkdir .pytest_reports
    del /q .pytest_reports\*.json
)

rem Distributed tests on Windows require PYTHON to carry an MPI launcher, as in
rem set "PYTHON=mpiexec -n 2 python" (see doc/sources/tests.rst). The check above
rem only recognizes the literal string "python", so any other bare interpreter -
rem conda-build passes %PREFIX%\python.exe - reached the block below and ran the
rem MPI steps single-rank with no MPI stack installed at all, since meta.yaml
rem marks mpi, mpi4py and pytest-mpi as "# [not win]". That looked like a pass
rem rather than an error: pytest rejects --with-mpi as an unknown argument when
rem pytest-mpi is missing, and helper_mpi_tests.py used to discard its status.
echo %PYTHON% | findstr /i /c:"mpiexec" /c:"mpirun" >nul
if errorlevel 1 set NO_DIST=1

echo "NO_DIST=%NO_DIST%"
setlocal enabledelayedexpansion
pytest %PYTEST_CONFIG% -s "%1tests" %PYTEST_ARGS:FILENAME=legacy_report% || set exitcode=1
pytest %PYTEST_CONFIG% --pyargs daal4py %PYTEST_ARGS:FILENAME=daal4py_report% || set exitcode=1
pytest %PYTEST_CONFIG% --pyargs sklearnex %PYTEST_ARGS:FILENAME=sklearnex_report% || set exitcode=1
pytest %PYTEST_CONFIG% --pyargs onedal %PYTEST_ARGS:FILENAME=onedal_report% || set exitcode=1
pytest %PYTEST_CONFIG% "%1.ci\scripts\test_global_patch.py" %PYTEST_ARGS:FILENAME=global_patching_report% || set exitcode=1
if NOT "%NO_DIST%"=="1" (
    %PYTHON% "%1tests\helper_mpi_tests.py"^
        pytest -k spmd --with-mpi %PYTEST_CONFIG% -s --pyargs sklearnex %PYTEST_ARGS:FILENAME=sklearnex_spmd%
    if !errorlevel! NEQ 0 (
        set exitcode=1
    )
    %PYTHON% "%1tests\helper_mpi_tests.py"^
        pytest --with-mpi %PYTEST_CONFIG% -s "%1tests\test_daal4py_spmd_examples.py" %PYTEST_ARGS:FILENAME=mpi_legacy%
    if !errorlevel! NEQ 0 (
        set exitcode=1
    )
    %PYTHON% "%1tests\helper_mpi_tests.py"^
        pytest --with-mpi %PYTEST_CONFIG% -s "%1tests\test_mpi_lifecycle.py" %PYTEST_ARGS:FILENAME=mpi_lifecycle%
    if !errorlevel! NEQ 0 (
        set exitcode=1
    )
    rem Not a pytest module, so there is nothing for the helper above to launch,
    rem and it could not be one: this covers the daal4py-owned MPI path, which
    rem requires that nothing has initialized MPI before daal4py does, while the
    rem helper and pytest-mpi both import mpi4py, which initializes it. PYTHON
    rem carries the launcher here, as it does for the steps above, so it also
    rem supplies the more than one rank this needs.
    %PYTHON% "%1tests\mpi_lifecycle_smoke.py"
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

@echo off
rem ==============================================================================
rem Copyright Contributors to the oneDAL Project
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
rem ==============================================================================

rem Per-output packager for the scikit-learn-intelex conda split (Windows).
rem Mirrors pack.sh: top-level bld.bat staged the install under
rem %SRC_DIR%\__sklearnex_stage; this script copies the subset belonging to
rem the current %PKG_NAME% into %PREFIX%.

setlocal EnableDelayedExpansion

IF "%SKLEARNEX_STAGE%"=="" set "SKLEARNEX_STAGE=%SRC_DIR%\__sklearnex_stage"

IF NOT EXIST "%SKLEARNEX_STAGE%" (
    echo pack.bat: staging root not found at %SKLEARNEX_STAGE% 1>&2
    echo pack.bat: expected top-level bld.bat to have populated it via 'setup.py install --root %%SKLEARNEX_STAGE%%' 1>&2
    exit /b 1
)

rem Locate the staged onedal package directory.
set "STAGED_ONEDAL="
for /f "delims=" %%i in ('dir /s /b /ad "%SKLEARNEX_STAGE%\onedal" 2^>nul ^| findstr /i "\\Lib\\site-packages\\onedal$"') do (
    set "STAGED_ONEDAL=%%i"
    goto :found
)
:found
IF "%STAGED_ONEDAL%"=="" (
    echo pack.bat: staged onedal package not found under %SKLEARNEX_STAGE% 1>&2
    exit /b 1
)

rem Derive the staged sys.prefix: ...\<sys.prefix>\Lib\site-packages\onedal
rem -> strip the trailing "\Lib\site-packages\onedal".
set "STAGED_PREFIX=!STAGED_ONEDAL:\Lib\site-packages\onedal=!"

IF /I "%PKG_NAME%"=="scikit-learn-intelex" (
    rem Copy everything from the staged sys.prefix into %PREFIX%, then strip
    rem the DPC backend .pyd files — they belong to scikit-learn-intelex-gpu.
    xcopy /e /q /y /i "%STAGED_PREFIX%\*" "%PREFIX%\" >nul
    if errorlevel 1 exit /b 1
    set "TARGET_ONEDAL=%PREFIX%\Lib\site-packages\onedal"
    del /q "!TARGET_ONEDAL!\_onedal_py_dpc*" 2>nul
    del /q "!TARGET_ONEDAL!\_onedal_py_spmd_dpc*" 2>nul
    exit /b 0
)

IF /I "%PKG_NAME%"=="scikit-learn-intelex-gpu" (
    rem Copy ONLY the DPC backend .pyd files into onedal/.
    set "TARGET_ONEDAL=%PREFIX%\Lib\site-packages\onedal"
    mkdir "!TARGET_ONEDAL!" 2>nul
    set "FOUND_DPC=0"
    for %%f in ("!STAGED_ONEDAL!\_onedal_py_dpc*") do (
        copy /y "%%f" "!TARGET_ONEDAL!\" >nul
        set "FOUND_DPC=1"
    )
    if "!FOUND_DPC!"=="0" (
        echo pack.bat: no DPC backend .pyd files found in !STAGED_ONEDAL! 1>&2
        exit /b 1
    )
    exit /b 0
)

echo pack.bat: unknown PKG_NAME='%PKG_NAME%' 1>&2
exit /b 1

@echo off
rem ============================================================================
rem Copyright 2024 Intel Corporation
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

rem %1 - dpcpp activate flag

rem prepare vc
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall" x64
rem prepare TBB + MKL from oneAPI if a parameter is given.
IF "%1"=="" (
   call .\oneapi\setvars.bat
) ELSE (
   call .\oneapi\tbb\latest\env\vars.bat
   call .\oneapi\mkl\latest\env\vars.bat
)
rem prepare oneDAL
call .\__release_win\daal\latest\env\vars.bat

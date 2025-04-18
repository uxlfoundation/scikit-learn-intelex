#! /usr/bin/env python
# ===============================================================================
# Copyright 2021 Intel Corporation
# Copyright 2024 Fujitsu Limited
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
# ===============================================================================

import logging
import os
import platform as plt
import subprocess
import sys
from os.path import join as jp
from sysconfig import get_config_var, get_paths

import numpy as np

logger = logging.getLogger("sklearnex")

IS_WIN = False
IS_MAC = False
IS_LIN = False

if "linux" in sys.platform:
    IS_LIN = True
elif sys.platform == "darwin":
    IS_MAC = True
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True


def custom_build_cmake_clib(
    iface,
    onedal_major_binary_version=1,
    no_dist=True,
    mpi_root=None,
    use_parameters_lib=True,
    use_abs_rpath=False,
    use_gcov=False,
    n_threads=1,
):
    import pybind11

    root_dir = os.path.normpath(jp(os.path.dirname(__file__), ".."))
    logger.info(f"Project directory is: {root_dir}")

    builder_directory = jp(root_dir, "scripts")
    abs_build_temp_path = jp(root_dir, "build", f"backend_{iface}")
    install_directory = jp(root_dir, "onedal")
    logger.info(f"Builder directory: {builder_directory}")
    logger.info(f"Install directory: {install_directory}")

    cmake_generator = "-GNinja" if IS_WIN else ""
    python_include = get_paths()["include"]
    win_python_path_lib = os.path.abspath(jp(get_config_var("LIBDEST"), "..", "libs"))
    python_library_dir = win_python_path_lib if IS_WIN else get_config_var("LIBDIR")
    numpy_include = np.get_include()

    cxx = os.getenv("CXX")
    if iface in ["dpc", "spmd_dpc"]:
        default_dpc_compiler = "icx" if IS_WIN else "icpx"
        if not cxx:
            cxx = default_dpc_compiler
        elif not (default_dpc_compiler in cxx):
            logger.warning(
                "Trying to build DPC module with a potentially non-DPC-capable compiler. Will forcefully change compiler to ICX."
            )
            cxx = default_dpc_compiler

    build_distribute = iface == "spmd_dpc" and not no_dist and IS_LIN

    logger.info(f"Build DPCPP SPMD functionality: {str(build_distribute)}")

    if build_distribute:
        MPI_INCDIRS = jp(mpi_root, "include")
        MPI_LIBDIRS = jp(mpi_root, "lib")
        MPI_LIBNAME = getattr(os.environ, "MPI_LIBNAME", None)
        if MPI_LIBNAME:
            MPI_LIBS = MPI_LIBNAME
        elif IS_WIN:
            if os.path.isfile(jp(mpi_root, "lib", "mpi.lib")):
                MPI_LIBS = "mpi"
            if os.path.isfile(jp(mpi_root, "lib", "impi.lib")):
                MPI_LIBS = "impi"
            assert MPI_LIBS, "Couldn't find MPI library"
        else:
            MPI_LIBS = "mpi"

    arch_dir = plt.machine()
    plt_dict = {"x86_64": "intel64", "AMD64": "intel64", "aarch64": "arm"}
    arch_dir = plt_dict[arch_dir] if arch_dir in plt_dict else arch_dir
    use_parameters_arg = "yes" if use_parameters_lib else "no"
    logger.info(f"Build using parameters library: {use_parameters_arg}")

    # Note: this uses env. variable 'CXX' instead of option 'CMAKE_CXX_COMPILER',
    # in order to propagate both potential user-passed arguments and flags, such as:
    #     CXX="ccache icpx"
    #     CXX="icpx -O0"
    env_build = dict(os.environ)
    if cxx:
        env_build["CXX"] = cxx
    cmake_args = [
        "cmake",
        cmake_generator,
        "-S" + builder_directory,
        "-B" + abs_build_temp_path,
        "-DCMAKE_INSTALL_PREFIX=" + install_directory,
        "-DCMAKE_PREFIX_PATH=" + install_directory,
        "-DIFACE=" + iface,
        "-DONEDAL_MAJOR_BINARY=" + str(onedal_major_binary_version),
        "-DPYTHON_INCLUDE_DIR=" + python_include,
        "-DNUMPY_INCLUDE_DIRS=" + numpy_include,
        "-DPYTHON_LIBRARY_DIR=" + python_library_dir,
        "-DoneDAL_INCLUDE_DIRS=" + jp(os.environ["DALROOT"], "include"),
        "-DoneDAL_LIBRARY_DIR=" + jp(os.environ["DALROOT"], "lib", arch_dir),
        "-Dpybind11_DIR=" + pybind11.get_cmake_dir(),
        "-DoneDAL_USE_PARAMETERS_LIB=" + use_parameters_arg,
    ]

    if build_distribute:
        cmake_args += [
            "-DMPI_INCLUDE_DIRS=" + MPI_INCDIRS,
            "-DMPI_LIBRARY_DIR=" + MPI_LIBDIRS,
            "-DMPI_LIBS=" + MPI_LIBS,
        ]

    if use_abs_rpath:
        cmake_args += ["-DADD_ONEDAL_RPATH=ON"]

    if use_gcov:
        cmake_args += ["-DSKLEARNEX_GCOV=ON"]

    # the number of parallel processes is dictated by MAKEFLAGS (see setup.py)
    # using make conventions (i.e. -j flag) but is set as a cmake argument to
    # support Windows and Linux simultaneously
    make_args = ["cmake", "--build", abs_build_temp_path, "-j" + str(n_threads)]

    make_install_args = [
        "cmake",
        "--install",
        abs_build_temp_path,
    ]

    subprocess.check_call(cmake_args, env=env_build)
    subprocess.check_call(make_args, env=env_build)
    subprocess.check_call(make_install_args, env=env_build)

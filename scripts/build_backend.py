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
import multiprocessing
import os
import platform as plt
import subprocess
import sys
from math import floor
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
    cxx=None,
    onedal_major_binary_version=1,
    no_dist=True,
    use_parameters_lib=True,
    use_abs_rpath=False,
    use_gcov=False,
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

    if iface in ["dpc", "spmd_dpc"]:
        if IS_WIN:
            cxx = "icx"
        else:
            cxx = "icpx"
    elif cxx is None:
        raise RuntimeError("CXX compiler shall be specified")

    build_distribute = iface == "spmd_dpc" and not no_dist and IS_LIN

    logger.info(f"Build DPCPP SPMD functionality: {str(build_distribute)}")

    if build_distribute:
        mpi_root = os.environ["MPIROOT"]
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

    cmake_args = [
        "cmake",
        cmake_generator,
        "-S" + builder_directory,
        "-B" + abs_build_temp_path,
        "-DCMAKE_CXX_COMPILER=" + cxx,
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

    cpu_count = multiprocessing.cpu_count()
    # limit parallel cmake jobs if memory size is insufficient
    # TODO: add on all platforms
    if IS_LIN:
        with open("/proc/meminfo", "r") as meminfo_file_obj:
            memfree = meminfo_file_obj.read().split("\n")[1].split(" ")
            while "" in memfree:
                memfree.remove("")
            memfree = int(memfree[1])  # total memory in kB
        cpu_count = min(cpu_count, floor(max(1, memfree / 2**20)))

    make_args = ["cmake", "--build", abs_build_temp_path, "-j " + str(cpu_count)]

    make_install_args = [
        "cmake",
        "--install",
        abs_build_temp_path,
    ]

    subprocess.check_call(cmake_args)
    subprocess.check_call(make_args)
    subprocess.check_call(make_install_args)

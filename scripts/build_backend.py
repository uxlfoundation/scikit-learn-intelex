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


def _get_required_onedal_libraries(
    iface, major_version, use_parameters_lib=True, is_win=False, is_mac=False
):
    """Return the exact oneDAL libraries required by the selected backend."""
    if iface not in ("host", "dpc", "spmd_dpc"):
        raise ValueError(f"Unsupported oneDAL backend: {iface}")

    is_dpc = iface in ("dpc", "spmd_dpc")
    backend_name = "onedal_dpc" if is_dpc else "onedal"
    library_names = [backend_name, "onedal_core"]
    if not is_win:
        library_names.append("onedal_thread")
    if use_parameters_lib:
        if is_win:
            parameter_name = (
                "onedal_core_parameters_dpc_dll"
                if is_dpc
                else "onedal_core_parameters_dll"
            )
        else:
            parameter_name = "onedal_parameters_dpc" if is_dpc else "onedal_parameters"
        library_names.append(parameter_name)

    if is_win:
        library_names = [
            name if name.endswith("_dll") else f"{name}_dll" for name in library_names
        ]
        return [f"{name}.{major_version}.lib" for name in library_names]
    if is_mac:
        return [f"lib{name}.{major_version}.dylib" for name in library_names]
    return [f"lib{name}.so.{major_version}" for name in library_names]


def _get_onedal_library_dir(
    dal_root,
    arch_dir,
    iface="host",
    major_version=1,
    use_parameters_lib=True,
    is_win=False,
    is_mac=False,
):
    """Find a complete oneDAL library directory in packaged and classic layouts."""
    candidates = [jp(dal_root, "lib", arch_dir)]
    if is_win:
        candidates.append(jp(dal_root, "Library", "lib"))
    candidates.append(jp(dal_root, "lib"))
    required = _get_required_onedal_libraries(
        iface,
        major_version,
        use_parameters_lib=use_parameters_lib,
        is_win=is_win,
        is_mac=is_mac,
    )

    checked = []
    for candidate in candidates:
        missing = [name for name in required if not os.path.isfile(jp(candidate, name))]
        if not missing:
            return candidate
        checked.append(f"{candidate} (missing: {', '.join(missing)})")

    raise FileNotFoundError(
        "Could not find a complete oneDAL library set: " + "; ".join(checked)
    )


def custom_build_cmake_clib(
    iface,
    onedal_major_binary_version=1,
    no_dist=True,
    mpi_root=None,
    use_parameters_lib: bool = True,
    use_abs_rpath: bool = False,
    use_gcov: bool = False,
    n_threads: int = 1,
    is_win: bool = False,
    is_lin: bool = False,
    debug_build: bool = False,
    using_lld: bool = False,
):
    import pybind11

    root_dir = os.path.normpath(jp(os.path.dirname(__file__), ".."))
    logger.info(f"Project directory is: {root_dir}")

    builder_directory = jp(root_dir, "scripts")
    abs_build_temp_path = jp(root_dir, "build", f"backend_{iface}")
    install_directory = jp(root_dir, "onedal")
    logger.info(f"Builder directory: {builder_directory}")
    logger.info(f"Install directory: {install_directory}")

    cmake_generator = "-GNinja" if is_win else ""
    python_include = get_paths()["include"]
    win_python_path_lib = os.path.abspath(jp(get_config_var("LIBDEST"), "..", "libs"))
    python_library_dir = win_python_path_lib if is_win else get_config_var("LIBDIR")
    numpy_include = np.get_include()

    cxx = os.getenv("CXX")
    if iface in ["dpc", "spmd_dpc"]:
        default_dpc_compiler = "icx" if is_win else "icpx"
        if not cxx:
            cxx = default_dpc_compiler
        elif not (default_dpc_compiler in cxx):
            logger.warning(
                "Trying to build DPC module with a potentially non-DPC-capable compiler. Will forcefully change compiler to ICX."
            )
            cxx = default_dpc_compiler

    build_distribute = iface == "spmd_dpc" and not no_dist and is_lin

    logger.info(f"Build DPCPP SPMD functionality: {str(build_distribute)}")

    if build_distribute:
        MPI_INCDIRS = jp(mpi_root, "include")
        MPI_LIBDIRS = jp(mpi_root, "lib")
        MPI_LIBNAME = os.environ.get("MPI_LIBNAME")
        if MPI_LIBNAME:
            MPI_LIBS = MPI_LIBNAME
        elif is_win:
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
    onedal_library_dir = _get_onedal_library_dir(
        os.environ["DALROOT"],
        arch_dir,
        iface=iface,
        major_version=onedal_major_binary_version,
        use_parameters_lib=use_parameters_lib,
        is_win=is_win,
        is_mac=plt.system() == "Darwin",
    )
    logger.info(f"oneDAL library directory: {onedal_library_dir}")
    use_parameters_arg = "yes" if use_parameters_lib else "no"
    logger.info(f"Build using parameters library: {use_parameters_arg}")

    # Note: this uses env. variable 'CXX' instead of option 'CMAKE_CXX_COMPILER',
    # in order to propagate both potential user-passed arguments and flags, such as:
    #     CXX="ccache icpx"
    #     CXX="icpx -O0"
    env_build = dict(os.environ)
    if cxx:
        env_build["CXX"] = cxx
    sanitizer = os.environ.get("SKLEARNEX_SANITIZER", "")
    if sanitizer and sanitizer not in ("address", "undefined", "thread"):
        raise ValueError(f"Unsupported sanitizer: {sanitizer}")
    build_type = "Debug" if debug_build else "RelWithDebInfo" if sanitizer else "Release"
    free_threading = bool(get_config_var("Py_GIL_DISABLED"))

    cmake_args = ["cmake"]
    if cmake_generator:
        cmake_args.append(cmake_generator)
    cmake_args += [
        "-S" + builder_directory,
        "-B" + abs_build_temp_path,
        "-DCMAKE_INSTALL_PREFIX=" + install_directory,
        "-DCMAKE_PREFIX_PATH=" + install_directory,
        "-DIFACE=" + iface,
        "-DONEDAL_MAJOR_BINARY=" + str(onedal_major_binary_version),
        "-DPYTHON_EXECUTABLE=" + sys.executable,
        "-DPython_EXECUTABLE=" + sys.executable,
        "-DPYTHON_INCLUDE_DIR=" + python_include,
        "-DNUMPY_INCLUDE_DIRS=" + numpy_include,
        "-DPYTHON_LIBRARY_DIR=" + python_library_dir,
        "-DoneDAL_INCLUDE_DIRS=" + jp(os.environ["DALROOT"], "include"),
        "-DoneDAL_LIBRARY_DIR=" + onedal_library_dir,
        "-Dpybind11_DIR=" + pybind11.get_cmake_dir(),
        "-DoneDAL_USE_PARAMETERS_LIB=" + use_parameters_arg,
        f"-DUSING_LLD={'ON' if using_lld else 'OFF'}",
        f"-DCMAKE_BUILD_TYPE={build_type}",
        f"-DSKLEARNEX_FREE_THREADING={'ON' if free_threading else 'OFF'}",
    ]

    # Guard against CMake selecting an ABI-incompatible interpreter (for example,
    # cp314 instead of cp314t) even when both are installed on the build host.
    python_soabi = get_config_var("SOABI")
    if python_soabi:
        cmake_args += ["-DEXPECTED_PYTHON_SOABI=" + python_soabi]

    if sanitizer:
        cmake_args += [f"-DSKLEARNEX_SANITIZER={sanitizer}"]

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
    # Keep the job count as a separate argument: CMake 3.13 accepts ``-j 2``
    # but not the concatenated ``-j2`` spelling.
    make_args = ["cmake", "--build", abs_build_temp_path, "-j", str(n_threads)]

    # ``cmake --install`` was added in CMake 3.15. Use the generated install
    # target so the legacy GIL-enabled path remains compatible with CMake 3.13.
    make_install_args = [
        "cmake",
        "--build",
        abs_build_temp_path,
        "--target",
        "install",
    ]

    subprocess.check_call(cmake_args, env=env_build)
    subprocess.check_call(make_args, env=env_build)
    subprocess.check_call(make_install_args, env=env_build)

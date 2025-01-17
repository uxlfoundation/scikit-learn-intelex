#! /usr/bin/env python
# ==============================================================================
# Copyright 2014 Intel Corporation
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
# ==============================================================================

import glob

# System imports
import os
import pathlib
import platform as plt
import re
import shutil
import sys
import time
from ctypes.util import find_library
from os.path import join as jp
from sysconfig import get_config_vars

import numpy as np
import setuptools.command.build as orig_build
import setuptools.command.develop as orig_develop
from Cython.Build import cythonize
from setuptools import Extension, setup

import scripts.build_backend as build_backend
from scripts.package_helpers import get_packages_with_tests
from scripts.version import get_onedal_shared_libs, get_onedal_version

USE_ABS_RPATH = False
ARG_ABS_RPATH = "--abs-rpath"
if ARG_ABS_RPATH in sys.argv:
    USE_ABS_RPATH = True
    sys.argv = [arg for arg in sys.argv if arg != ARG_ABS_RPATH]

IS_WIN = False
IS_MAC = False
IS_LIN = False

dal_root = os.environ.get("DALROOT")

arch_dir = plt.machine()
plt_dict = {"x86_64": "intel64", "AMD64": "intel64", "aarch64": "arm"}
arch_dir = plt_dict[arch_dir] if arch_dir in plt_dict else arch_dir
if dal_root is None:
    raise RuntimeError("Not set DALROOT variable")

if "linux" in sys.platform:
    IS_LIN = True
    lib_dir = jp(dal_root, "lib", arch_dir)
elif sys.platform == "darwin":
    IS_MAC = True
    lib_dir = jp(dal_root, "lib")
elif sys.platform in ["win32", "cygwin"]:
    IS_WIN = True
    lib_dir = jp(dal_root, "lib", arch_dir)
else:
    assert False, sys.platform + " not supported"

ONEDAL_MAJOR_BINARY_VERSION, ONEDAL_MINOR_BINARY_VERSION = get_onedal_version(
    dal_root, "binary"
)
ONEDAL_VERSION = get_onedal_version(dal_root)
ONEDAL_2021_3 = 2021 * 10000 + 3 * 100
ONEDAL_2023_0_1 = 2023 * 10000 + 0 * 100 + 1
is_onedal_iface = (
    os.environ.get("OFF_ONEDAL_IFACE", "0") == "0" and ONEDAL_VERSION >= ONEDAL_2021_3
)

sklearnex_version = (
    os.environ["SKLEARNEX_VERSION"]
    if "SKLEARNEX_VERSION" in os.environ
    else time.strftime("%Y%m%d.%H%M%S")
)

trues = ["true", "True", "TRUE", "1", "t", "T", "y", "Y", "Yes", "yes", "YES"]
no_dist = True if "NO_DIST" in os.environ and os.environ["NO_DIST"] in trues else False
no_dpc = True if "NO_DPC" in os.environ and os.environ["NO_DPC"] in trues else False
no_stream = "NO_STREAM" in os.environ and os.environ["NO_STREAM"] in trues
use_gcov = "SKLEARNEX_GCOV" in os.environ and os.environ["SKLEARNEX_GCOV"] in trues
debug_build = os.getenv("DEBUG_BUILD") == "1"
mpi_root = None if no_dist else os.environ["MPIROOT"]
dpcpp = (
    shutil.which("icpx") is not None
    and "onedal_dpc" in get_onedal_shared_libs(dal_root)
    and not no_dpc
    and not (IS_WIN and debug_build)
)

use_parameters_lib = (not IS_WIN) and (ONEDAL_VERSION >= 20240000)

build_distribute = dpcpp and not no_dist and IS_LIN

daal_lib_dir = lib_dir if (IS_MAC or os.path.isdir(lib_dir)) else os.path.dirname(lib_dir)
ONEDAL_LIBDIRS = [daal_lib_dir]
if IS_WIN:
    ONEDAL_LIBDIRS.append(f"{os.environ.get('CONDA_PREFIX')}/Library/lib")

if no_stream:
    print("\nDisabling support for streaming mode\n")
if no_dist:
    print("\nDisabling support for distributed mode\n")
    DIST_CFLAGS = []
    DIST_CPPS = []
    MPI_INCDIRS = []
    MPI_LIBDIRS = []
    MPI_LIBS = []
    MPI_CPPS = []
else:
    DIST_CFLAGS = [
        "-D_DIST_",
    ]
    DIST_CPPS = ["src/transceiver.cpp"]
    MPI_INCDIRS = [jp(mpi_root, "include")]
    MPI_LIBDIRS = [jp(mpi_root, "lib")]
    MPI_LIBNAME = getattr(os.environ, "MPI_LIBNAME", None)
    if MPI_LIBNAME:
        MPI_LIBS = [MPI_LIBNAME]
    elif IS_WIN:
        if os.path.isfile(jp(mpi_root, "lib", "mpi.lib")):
            MPI_LIBS = ["mpi"]
        if os.path.isfile(jp(mpi_root, "lib", "impi.lib")):
            MPI_LIBS = ["impi"]
        assert MPI_LIBS, "Couldn't find MPI library"
    else:
        MPI_LIBS = ["mpi"]
    MPI_CPPS = ["src/mpi/mpi_transceiver.cpp"]


def get_sdl_cflags():
    if IS_LIN or IS_MAC:
        return DIST_CFLAGS + [
            "-fstack-protector-strong",
            "-fPIC",
            "-D_FORTIFY_SOURCE=2",
            "-Wformat",
            "-Wformat-security",
            "-fno-strict-overflow",
            "-fno-delete-null-pointer-checks",
        ]
    if IS_WIN:
        return DIST_CFLAGS + ["-GS"]


def get_sdl_ldflags():
    if IS_LIN:
        return [
            "-Wl,-z,noexecstack,-z,relro,-z,now,-fstack-protector-strong,"
            "-fno-strict-overflow,-fno-delete-null-pointer-checks,-fwrapv"
        ]
    if IS_MAC:
        return [
            "-fstack-protector-strong",
            "-fno-strict-overflow",
            "-fno-delete-null-pointer-checks",
            "-fwrapv",
        ]
    if IS_WIN:
        return ["-NXCompat", "-DynamicBase"]


def get_daal_type_defines():
    daal_type_defines = [
        "DAAL_ALGORITHM_FP_TYPE",
        "DAAL_SUMMARY_STATISTICS_TYPE",
        "DAAL_DATA_TYPE",
    ]
    return [(d, "double") for d in daal_type_defines]


def get_libs(iface="daal"):
    major_version = ONEDAL_MAJOR_BINARY_VERSION
    if IS_WIN:
        libraries_plat = [f"onedal_core_dll.{major_version}"]
        onedal_lib = [
            f"onedal_dll.{major_version}",
        ]
        onedal_dpc_lib = [
            f"onedal_dpc_dll.{major_version}",
        ]
        if use_parameters_lib:
            onedal_lib += [
                f"onedal_parameters.{major_version}",
                f"onedal_parameters_dll.{major_version}",
            ]
            onedal_dpc_lib += [
                f"onedal_parameters_dpc_dll.{major_version}",
            ]
    elif IS_MAC:
        libraries_plat = [
            f"onedal_core.{major_version}",
            f"onedal_thread.{major_version}",
        ]
        onedal_lib = [
            f"onedal.{major_version}",
        ]
        onedal_dpc_lib = [
            f"onedal_dpc.{major_version}",
        ]
        if use_parameters_lib:
            onedal_lib += [
                f"onedal_parameters.{major_version}",
            ]
            onedal_dpc_lib += [
                f"onedal_parameters_dpc.{major_version}",
            ]
    else:
        libraries_plat = [
            f":libonedal_core.so.{major_version}",
            f":libonedal_thread.so.{major_version}",
        ]
        onedal_lib = [
            f":libonedal.so.{major_version}",
        ]
        onedal_dpc_lib = [
            f":libonedal_dpc.so.{major_version}",
        ]
        if use_parameters_lib:
            onedal_lib += [
                f":libonedal_parameters.so.{major_version}",
            ]
            onedal_dpc_lib += [
                f":libonedal_parameters_dpc.so.{major_version}",
            ]
    if iface == "onedal":
        libraries_plat = onedal_lib + libraries_plat
    elif iface == "onedal_dpc":
        libraries_plat = onedal_dpc_lib + libraries_plat
    return libraries_plat


def get_build_options():
    include_dir_plat = [
        os.path.abspath("./src"),
        os.path.abspath("."),
    ]
    include_dir_candidates = [
        jp(dal_root, "include"),
        jp(dal_root, "include", "dal"),
        jp(dal_root, "Library", "include", "dal"),
    ]
    for candidate in include_dir_candidates:
        if os.path.isdir(candidate):
            include_dir_plat.append(candidate)
    # FIXME it is a wrong place for this dependency
    if not no_dist:
        include_dir_plat.append(mpi_root + "/include")
    using_intel = os.environ.get("cc", "") in [
        "icc",
        "icpc",
        "icl",
        "dpcpp",
        "icx",
        "icpx",
    ]
    eca = [
        "-DPY_ARRAY_UNIQUE_SYMBOL=daal4py_array_API",
        '-DD4P_VERSION="' + sklearnex_version + '"',
        "-DNPY_ALLOW_THREADS=1",
    ]
    ela = []

    if using_intel and IS_WIN:
        include_dir_plat.append(
            jp(os.environ.get("ICPP_COMPILER16", ""), "compiler", "include")
        )
        eca += ["-std=c++17", "-w", "/MD"]
    elif not using_intel and IS_WIN:
        eca += ["-wd4267", "-wd4244", "-wd4101", "-wd4996", "/std:c++17"]
    else:
        eca += [
            "-std=c++17",
            "-w",
        ]  # '-D_GLIBCXX_USE_CXX11_ABI=0']

    # Security flags
    eca += get_sdl_cflags()
    ela += get_sdl_ldflags()

    if IS_MAC:
        eca.append("-stdlib=libc++")
        ela.append("-stdlib=libc++")
        ela.append("-Wl,-rpath,{}".format(daal_lib_dir))
        ela.append("-Wl,-rpath,@loader_path/../../../")
    elif IS_WIN:
        ela.append("-IGNORE:4197")
    elif IS_LIN and not any(
        x in os.environ and "-g" in os.environ[x]
        for x in ["CPPFLAGS", "CFLAGS", "LDFLAGS"]
    ):
        ela.append("-s")
    if IS_LIN:
        ela.append("-fPIC")
        ela.append(
            f"-Wl,-rpath,{(daal_lib_dir + ':') if USE_ABS_RPATH else ''}$ORIGIN/../../../"
        )
    return eca, ela, include_dir_plat


def getpyexts():
    eca, ela, include_dir_plat = get_build_options()
    libraries_plat = get_libs("daal")

    exts = []

    ext = Extension(
        "daal4py._daal4py",
        [
            os.path.abspath("src/daal4py.cpp"),
            os.path.abspath("build/daal4py_cpp.cpp"),
            os.path.abspath("build/daal4py_cy.pyx"),
        ]
        + DIST_CPPS,
        depends=glob.glob(jp(os.path.abspath("src"), "*.h")),
        include_dirs=include_dir_plat + [np.get_include()],
        extra_compile_args=eca,
        define_macros=get_daal_type_defines(),
        extra_link_args=ela,
        libraries=libraries_plat,
        library_dirs=ONEDAL_LIBDIRS,
        language="c++",
    )

    exts.extend(cythonize(ext))

    if not no_dist:
        mpi_include_dir = include_dir_plat + [np.get_include()] + MPI_INCDIRS
        mpi_depens = glob.glob(jp(os.path.abspath("src"), "*.h"))
        mpi_extra_link = ela + ["-Wl,-rpath,{}".format(x) for x in MPI_LIBDIRS]
        exts.append(
            Extension(
                "daal4py.mpi_transceiver",
                MPI_CPPS,
                depends=mpi_depens,
                include_dirs=mpi_include_dir,
                extra_compile_args=eca,
                define_macros=get_daal_type_defines(),
                extra_link_args=mpi_extra_link,
                libraries=libraries_plat + MPI_LIBS,
                library_dirs=ONEDAL_LIBDIRS + MPI_LIBDIRS,
                language="c++",
            )
        )
    return exts


cfg_vars = get_config_vars()
for key, value in get_config_vars().items():
    if isinstance(value, str):
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "").replace("-DNDEBUG", "")


def gen_pyx(odir):
    gtr_files = glob.glob(jp(os.path.abspath("generator"), "*")) + ["./setup.py"]
    src_files = [
        os.path.abspath("build/daal4py_cpp.h"),
        os.path.abspath("build/daal4py_cpp.cpp"),
        os.path.abspath("build/daal4py_cy.pyx"),
    ]
    if all(os.path.isfile(x) for x in src_files):
        src_files.sort(key=os.path.getmtime)
        gtr_files.sort(key=os.path.getmtime, reverse=True)
        if os.path.getmtime(src_files[0]) > os.path.getmtime(gtr_files[0]):
            print(
                "Generated files are all newer than generator code."
                "Skipping code generation"
            )
            return

    from generator.gen_daal4py import gen_daal4py

    odir = os.path.abspath(odir)
    if not os.path.isdir(odir):
        os.mkdir(odir)
    gen_daal4py(dal_root, odir, sklearnex_version, no_dist=no_dist, no_stream=no_stream)


gen_pyx(os.path.abspath("./build"))


def get_onedal_py_libs():
    ext_suffix = get_config_vars("EXT_SUFFIX")[0]
    libs = [f"_onedal_py_host{ext_suffix}", f"_onedal_py_dpc{ext_suffix}"]
    if build_distribute:
        libs += [f"_onedal_py_spmd_dpc{ext_suffix}"]
    if IS_WIN:
        ext_suffix_lib = ext_suffix.replace(".dll", ".lib")
        libs += [f"_onedal_py_host{ext_suffix_lib}", f"_onedal_py_dpc{ext_suffix_lib}"]
        if build_distribute:
            libs += [f"_onedal_py_spmd_dpc{ext_suffix_lib}"]
    return libs


class onedal_build:

    def run(self):
        self.onedal_run()
        super(onedal_build, self).run()
        self.onedal_post_build()

    def onedal_run(self):
        n_threads = self.parallel
        makeflags = os.getenv("MAKEFLAGS", "")
        # True is used by setuptools to indicate cpu_count for `parallel`
        # None is default for setuptools for single threading
        # take the last defined value in MAKEFLAGS, as it will be the one
        # used by cmake/make. Do regex in reverse to deal with missing values
        # and last values simultaneously in a simple fashion
        regex_inv = r"(?<!\S)\d*(?=j-(?!\S))|$"
        orig_n_threads = re.findall(regex_inv, makeflags[::-1])[0][::-1]

        if n_threads is None:
            n_threads = int(orig_n_threads) if orig_n_threads else os.cpu_count() or 1
        elif n_threads is True:
            n_threads = os.cpu_count() or 1

        cxx = os.getenv("CXX", "cl" if IS_WIN else "g++")
        build_onedal = lambda iface: build_backend.custom_build_cmake_clib(
            iface=iface,
            cxx=cxx,
            onedal_major_binary_version=ONEDAL_MAJOR_BINARY_VERSION,
            no_dist=no_dist,
            use_parameters_lib=use_parameters_lib,
            use_abs_rpath=USE_ABS_RPATH,
            use_gcov=use_gcov,
            n_threads=n_threads,
        )
        if is_onedal_iface:
            build_onedal("host")
            if dpcpp:
                build_onedal("dpc")
                if build_distribute:
                    build_onedal("spmd_dpc")

    def onedal_post_build(self):
        if IS_MAC:
            import subprocess

            # manually fix incorrect install_name of oneDAL 2023.0.1 libs
            major_version = ONEDAL_MAJOR_BINARY_VERSION
            major_is_available = (
                find_library(f"libonedal_core.{major_version}.dylib") is not None
            )
            if major_is_available and ONEDAL_VERSION == ONEDAL_2023_0_1:
                extension_libs = list(pathlib.Path(".").glob("**/*darwin.so"))
                onedal_libs = ["onedal", "onedal_dpc", "onedal_core", "onedal_thread"]
                for ext_lib in extension_libs:
                    for onedal_lib in onedal_libs:
                        subprocess.call(
                            "/usr/bin/install_name_tool -change "
                            f"lib{onedal_lib}.dylib "
                            f"lib{onedal_lib}.{major_version}.dylib "
                            f"{ext_lib}".split(" "),
                            shell=False,
                        )


class develop(onedal_build, orig_develop.develop):
    parallel = None


class build(onedal_build, orig_build.build):
    pass


project_urls = {
    "Bug Tracker": "https://github.com/uxlfoundation/scikit-learn-intelex",
    "Documentation": "https://intelpython.github.io/daal4py/",
    "Source Code": "https://github.com/uxlfoundation/scikit-learn-intelex/daal4py",
}

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()

packages_with_tests = [
    "daal4py",
    "daal4py.mb",
    "daal4py.sklearn",
    "daal4py.sklearn.cluster",
    "daal4py.sklearn.decomposition",
    "daal4py.sklearn.ensemble",
    "daal4py.sklearn.linear_model",
    "daal4py.sklearn.manifold",
    "daal4py.sklearn.metrics",
    "daal4py.sklearn.neighbors",
    "daal4py.sklearn.monkeypatch",
    "daal4py.sklearn.svm",
    "daal4py.sklearn.utils",
    "daal4py.sklearn.model_selection",
    "onedal",
    "onedal.common",
    "onedal.covariance",
    "onedal.datatypes",
    "onedal.decomposition",
    "onedal.ensemble",
    "onedal.neighbors",
    "onedal.primitives",
    "onedal.svm",
    "onedal.utils",
    "sklearnex",
    "sklearnex.basic_statistics",
    "sklearnex.cluster",
    "sklearnex.covariance",
    "sklearnex.decomposition",
    "sklearnex.ensemble",
    "sklearnex.glob",
    "sklearnex.linear_model",
    "sklearnex.manifold",
    "sklearnex.metrics",
    "sklearnex.model_selection",
    "sklearnex.neighbors",
    "sklearnex.preview",
    "sklearnex.preview.covariance",
    "sklearnex.preview.decomposition",
    "sklearnex.svm",
    "sklearnex.utils",
]

if ONEDAL_VERSION >= 20230100:
    packages_with_tests += ["onedal.basic_statistics", "onedal.linear_model"]

if ONEDAL_VERSION >= 20230200:
    packages_with_tests += ["onedal.cluster"]

if build_distribute:
    packages_with_tests += [
        "onedal.spmd",
        "onedal.spmd.covariance",
        "onedal.spmd.decomposition",
        "onedal.spmd.ensemble",
        "sklearnex.spmd",
        "sklearnex.spmd.covariance",
        "sklearnex.spmd.decomposition",
        "sklearnex.spmd.ensemble",
    ]
    if ONEDAL_VERSION >= 20230100:
        packages_with_tests += [
            "onedal.spmd.basic_statistics",
            "onedal.spmd.linear_model",
            "onedal.spmd.neighbors",
            "sklearnex.spmd.basic_statistics",
            "sklearnex.spmd.linear_model",
            "sklearnex.spmd.neighbors",
        ]
    if ONEDAL_VERSION >= 20230200:
        packages_with_tests += ["onedal.spmd.cluster", "sklearnex.spmd.cluster"]

setup(
    name="scikit-learn-intelex",
    description="Intel(R) Extension for Scikit-learn is a "
    "seamless way to speed up your Scikit-learn application.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    author="Intel Corporation",
    version=sklearnex_version,
    url="https://github.com/uxlfoundation/scikit-learn-intelex",
    author_email="onedal.maintainers@intel.com",
    maintainer_email="onedal.maintainers@intel.com",
    project_urls=project_urls,
    cmdclass={"develop": develop, "build": build},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Other Audience",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
        "Topic :: System",
        "Topic :: Software Development",
    ],
    python_requires=">=3.9",
    install_requires=[
        "scikit-learn>=1.0",
        "numpy>=1.19.5 ; python_version <= '3.9'",
        "numpy>=1.21.6 ; python_version == '3.10'",
        "numpy>=1.23.5 ; python_version >= '3.11'",
    ],
    keywords=["machine learning", "scikit-learn", "data science", "data analytics"],
    packages=get_packages_with_tests(packages_with_tests),
    package_data={
        "onedal": get_onedal_py_libs(),
    },
    ext_modules=getpyexts(),
)

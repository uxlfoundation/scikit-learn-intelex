# ==============================================================================
# Copyright contributors to the oneDAL project
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

"""Unit tests for the oneDAL discovery and naming helpers in build_backend.py.

CI builds against real oneDAL installations in more than one layout - the
classic ``lib/<arch>`` tree, and the flattened one the documentation job gets
from pip - so a layout this file gets wrong is caught there too, as a build
failure. What a build cannot report is *which* directory was rejected and why,
or that setup.py and scripts/CMakeLists.txt still agree on what oneDAL is
called on Windows: a disagreement there surfaces as a link error a long way
from its cause. Those are what this file pins down.

The last test covers the CMake invocation itself rather than the naming helpers:
which build type and which interpreter the backend build is told to use, on both
the GIL-enabled and the free-threaded build.
"""

import importlib.util
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "build_backend.py"
SPEC = importlib.util.spec_from_file_location("sklearnex_build_backend", MODULE_PATH)
build_backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_backend)

_get_onedal_library_dir = build_backend._get_onedal_library_dir
get_onedal_arch_dir = build_backend.get_onedal_arch_dir
get_onedal_libraries = build_backend.get_onedal_libraries
get_onedal_library_filenames = build_backend.get_onedal_library_filenames

# Arbitrary, and deliberately not the version this checkout builds against: the
# helpers under test take the major version as an argument, so a hard-coded one
# leaking into them would go unnoticed if it matched the real one.
MAJOR_VERSION = 4

LINUX_HOST_LIBRARIES = (
    f"libonedal.so.{MAJOR_VERSION}",
    f"libonedal_parameters.so.{MAJOR_VERSION}",
    f"libonedal_core.so.{MAJOR_VERSION}",
    f"libonedal_thread.so.{MAJOR_VERSION}",
)
WINDOWS_HOST_LIBRARIES = (
    f"onedal_dll.{MAJOR_VERSION}.lib",
    f"onedal_core_parameters_dll.{MAJOR_VERSION}.lib",
    f"onedal_core_dll.{MAJOR_VERSION}.lib",
)


def _create_libraries(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


@pytest.mark.parametrize(
    "layout,arch_dir,is_win,libraries",
    [
        # Classic oneDAL install; also the layout CI itself uses.
        (("lib", "intel64"), "intel64", False, LINUX_HOST_LIBRARIES),
        # oneDAL names the potential Linux aarch64 directory `arm`.
        (("lib", "arm"), "arm", False, LINUX_HOST_LIBRARIES),
        # Conda, pip and some source installs flatten the arch directory away.
        (("lib",), "intel64", False, LINUX_HOST_LIBRARIES),
        # Windows conda packages put libraries under the Library prefix.
        (("Library", "lib"), "intel64", True, WINDOWS_HOST_LIBRARIES),
    ],
)
def test_get_onedal_library_dir_accepts_supported_layouts(
    tmp_path, layout, arch_dir, is_win, libraries
):
    library_dir = tmp_path.joinpath(*layout)
    _create_libraries(library_dir, libraries)

    assert _get_onedal_library_dir(
        tmp_path, arch_dir, major_version=MAJOR_VERSION, is_win=is_win
    ) == str(library_dir)


@pytest.mark.parametrize(
    "machine,arch_dir",
    [
        # platform.machine() spells the same architecture differently per OS:
        # x86_64/aarch64 on Linux and macOS, AMD64/ARM64 on Windows.
        ("x86_64", "intel64"),
        ("AMD64", "intel64"),
        ("aarch64", "arm"),
        ("ARM64", "arm"),
    ],
)
def test_get_onedal_arch_dir(machine, arch_dir):
    assert get_onedal_arch_dir(machine) == arch_dir


def test_get_onedal_library_dir_skips_incomplete_directory(tmp_path):
    # An arch directory left over from a different oneDAL major version must not
    # win over a complete one, which is what a plain isdir() check would do.
    flat = tmp_path / "lib"
    _create_libraries(flat, LINUX_HOST_LIBRARIES)
    _create_libraries(
        flat / "intel64",
        (
            f"libonedal.so.{MAJOR_VERSION - 1}",
            f"libonedal_core.so.{MAJOR_VERSION - 1}",
        ),
    )

    assert _get_onedal_library_dir(
        tmp_path, "intel64", major_version=MAJOR_VERSION
    ) == str(flat)


def test_get_onedal_library_dir_names_every_directory_and_miss(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        _get_onedal_library_dir(tmp_path, "intel64", major_version=MAJOR_VERSION)

    message = str(exc_info.value)
    assert str(tmp_path / "lib" / "intel64") in message
    assert str(tmp_path / "lib") in message
    assert f"libonedal_thread.so.{MAJOR_VERSION}" in message


@pytest.mark.parametrize("iface", ["daal", "host", "dpc", "spmd_dpc"])
@pytest.mark.parametrize(
    "is_win,is_mac,prefix,suffix",
    [
        (False, False, ":lib", f".so.{MAJOR_VERSION}"),
        (False, True, "", f".{MAJOR_VERSION}"),
        (True, False, "", f".{MAJOR_VERSION}"),
    ],
)
def test_linker_names_and_filenames_describe_the_same_libraries(
    iface, is_win, is_mac, prefix, suffix
):
    # setup.py links by these names and the CMake backend looks for the files
    # they resolve to. A divergence between the two only shows up as a link
    # error inside CI, so pin the correspondence here.
    linker_names = get_onedal_libraries(
        iface, MAJOR_VERSION, is_win=is_win, is_mac=is_mac, use_parameters_lib=not is_win
    )
    filenames = get_onedal_library_filenames(
        iface, MAJOR_VERSION, is_win=is_win, is_mac=is_mac, use_parameters_lib=not is_win
    )
    assert len(linker_names) == len(filenames)

    for linker_name, filename in zip(linker_names, filenames):
        assert linker_name.startswith(prefix)
        assert linker_name.endswith(suffix)
        stem = linker_name[len(prefix) : -len(suffix)]
        if is_win:
            assert filename == f"{stem}.{MAJOR_VERSION}.lib"
        elif is_mac:
            assert filename == f"lib{stem}.{MAJOR_VERSION}.dylib"
        else:
            assert filename == f"lib{stem}.so.{MAJOR_VERSION}"


def test_backend_libraries_extend_the_daal_library_set():
    # daal4py links the core libraries; the oneDAL backends add an interface
    # library on top. Nothing may drop a core library on the way.
    core = get_onedal_libraries("daal", MAJOR_VERSION)
    for iface, expected in (("host", "onedal"), ("dpc", "onedal_dpc")):
        libraries = get_onedal_libraries(iface, MAJOR_VERSION, use_parameters_lib=False)
        assert libraries[0] == f":lib{expected}.so.{MAJOR_VERSION}"
        assert set(core).issubset(libraries)


def test_windows_library_names_match_cmake():
    # scripts/CMakeLists.txt spells the Windows import libraries out by hand,
    # including the onedal_core_parameters_*_dll names that do not follow the
    # pattern of the others. Compare against it directly.
    cmake_text = (REPO_ROOT / "scripts" / "CMakeLists.txt").read_text()
    in_cmake = set(re.findall(r'"(onedal[A-Za-z_]*_dll)\.\$\{ONEDAL_MAJOR', cmake_text))

    from_python = set()
    for iface in ("host", "dpc"):
        for name in get_onedal_libraries(iface, MAJOR_VERSION, is_win=True):
            from_python.add(name.rsplit(".", 1)[0])

    assert from_python == in_cmake


def test_unsupported_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported oneDAL backend"):
        get_onedal_libraries("host_dpc", MAJOR_VERSION)


@pytest.mark.parametrize(
    "is_win,debug_build,expected_build_type",
    [
        (False, False, "Release"),
        (False, True, "Debug"),
        (True, False, "Release"),
        (True, True, "Debug"),
    ],
)
@pytest.mark.parametrize("free_threading", [False, True])
def test_cmake_build_type_is_explicit(
    monkeypatch,
    tmp_path,
    is_win,
    debug_build,
    expected_build_type,
    free_threading,
):
    dal_root = tmp_path / "dal"
    if is_win:
        library_dir = dal_root / "Library" / "lib"
        libraries = WINDOWS_HOST_LIBRARIES
    else:
        library_dir = dal_root / "lib" / "intel64"
        libraries = LINUX_HOST_LIBRARIES
    _create_libraries(library_dir, libraries)

    soabi = "cp312-win_amd64" if is_win else "cpython-312-x86_64-linux-gnu"
    monkeypatch.setenv("DALROOT", str(dal_root))
    monkeypatch.setitem(
        sys.modules, "pybind11", SimpleNamespace(get_cmake_dir=lambda: "pybind11-cmake")
    )
    monkeypatch.setattr(build_backend.np, "get_include", lambda: "numpy-include")
    monkeypatch.setattr(build_backend, "get_paths", lambda: {"include": "python-include"})
    monkeypatch.setattr(
        build_backend,
        "get_config_var",
        lambda name: {
            "LIBDEST": str(tmp_path / "python" / "Lib"),
            "LIBDIR": str(tmp_path / "python" / "lib"),
            "SOABI": soabi,
            "Py_GIL_DISABLED": int(free_threading),
        }.get(name),
    )

    calls = []
    monkeypatch.setattr(
        build_backend.subprocess,
        "check_call",
        lambda command, env: calls.append(command),
    )

    build_backend.custom_build_cmake_clib(
        "host",
        onedal_major_binary_version=MAJOR_VERSION,
        is_win=is_win,
        is_lin=not is_win,
        debug_build=debug_build,
    )

    build_type_args = [arg for arg in calls[0] if arg.startswith("-DCMAKE_BUILD_TYPE=")]
    assert build_type_args == [f"-DCMAKE_BUILD_TYPE={expected_build_type}"]
    assert f"-DSKLEARNEX_FREE_THREADING={'ON' if free_threading else 'OFF'}" in calls[0]
    assert "" not in calls[0]

    # The interpreter pinning is scoped to the free-threaded build, which is the
    # only path that calls find_package(Python) and so the only one that can
    # pick an ABI-incompatible interpreter. Passing it unconditionally would
    # change how the GIL-enabled build resolves Python for no reason.
    pinning_args = [
        f"-DPython_EXECUTABLE={sys.executable}",
        f"-DPython_ROOT_DIR={sys.prefix}",
        f"-DEXPECTED_PYTHON_SOABI={soabi}",
    ]
    for arg in pinning_args:
        assert (arg in calls[0]) is free_threading
    # The deprecated spelling must not be passed: pybind11's FindPython does not
    # read it, and setting it can steer the search away from Python_EXECUTABLE.
    assert not [arg for arg in calls[0] if arg.startswith("-DPYTHON_EXECUTABLE=")]
    assert ("-GNinja" in calls[0]) is is_win
    assert calls[1][:2] == ["cmake", "--build"]
    assert Path(calls[1][2]).parts[-2:] == ("build", "backend_host")
    assert calls[1][3:] == ["-j1"]
    assert calls[2][:2] == ["cmake", "--install"]
    assert Path(calls[2][2]).parts[-2:] == ("build", "backend_host")
    assert calls[2][3:] == []

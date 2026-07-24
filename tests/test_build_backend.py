# ==============================================================================
# Copyright 2026 Intel Corporation
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

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "build_backend.py"
SPEC = importlib.util.spec_from_file_location("sklearnex_build_backend", MODULE_PATH)
build_backend = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_backend)

_get_onedal_library_dir = build_backend._get_onedal_library_dir
_get_required_onedal_libraries = build_backend._get_required_onedal_libraries

LINUX_HOST_LIBRARIES = (
    "libonedal.so.4",
    "libonedal_core.so.4",
    "libonedal_thread.so.4",
    "libonedal_parameters.so.4",
)


def _create_libraries(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


def test_get_onedal_library_dir_prefers_complete_arch_layout(tmp_path):
    flat = tmp_path / "lib"
    arch = flat / "intel64"
    _create_libraries(flat, LINUX_HOST_LIBRARIES)
    _create_libraries(arch, LINUX_HOST_LIBRARIES)

    assert _get_onedal_library_dir(tmp_path, "intel64", major_version=4) == str(arch)


def test_get_onedal_library_dir_falls_back_from_incomplete_arch_layout(tmp_path):
    flat = tmp_path / "lib"
    arch = flat / "intel64"
    _create_libraries(flat, LINUX_HOST_LIBRARIES)
    _create_libraries(arch, ("libonedal.so.3", "libonedal_core.so.3"))

    assert _get_onedal_library_dir(tmp_path, "intel64", major_version=4) == str(flat)


def test_get_onedal_library_dir_accepts_windows_conda_layout(tmp_path):
    windows_lib = tmp_path / "Library" / "lib"
    _create_libraries(
        windows_lib,
        (
            "onedal_dll.4.lib",
            "onedal_core_dll.4.lib",
            "onedal_core_parameters_dll.4.lib",
        ),
    )

    assert _get_onedal_library_dir(
        tmp_path, "intel64", major_version=4, is_win=True
    ) == str(windows_lib)


@pytest.mark.parametrize(
    "iface,major_version,is_win,is_mac,expected",
    [
        (
            "dpc",
            3,
            False,
            False,
            [
                "libonedal_dpc.so.3",
                "libonedal_core.so.3",
                "libonedal_thread.so.3",
                "libonedal_parameters_dpc.so.3",
            ],
        ),
        (
            "dpc",
            4,
            False,
            False,
            [
                "libonedal_dpc.so.4",
                "libonedal_core.so.4",
                "libonedal_thread.so.4",
                "libonedal_parameters_dpc.so.4",
            ],
        ),
        (
            "host",
            4,
            False,
            True,
            [
                "libonedal.4.dylib",
                "libonedal_core.4.dylib",
                "libonedal_thread.4.dylib",
                "libonedal_parameters.4.dylib",
            ],
        ),
        (
            "spmd_dpc",
            3,
            True,
            False,
            [
                "onedal_dpc_dll.3.lib",
                "onedal_core_dll.3.lib",
                "onedal_core_parameters_dpc_dll.3.lib",
            ],
        ),
    ],
)
def test_required_onedal_libraries_are_platform_specific(
    iface, major_version, is_win, is_mac, expected
):
    assert (
        _get_required_onedal_libraries(iface, major_version, is_win=is_win, is_mac=is_mac)
        == expected
    )


def test_get_onedal_library_dir_reports_missing_sonames(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        _get_onedal_library_dir(tmp_path, "intel64", major_version=4)

    message = str(exc_info.value)
    assert str(tmp_path / "lib" / "intel64") in message
    assert "libonedal_thread.so.4" in message
    assert str(tmp_path / "lib") in message


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
        libraries = (
            "onedal_dll.4.lib",
            "onedal_core_dll.4.lib",
            "onedal_core_parameters_dll.4.lib",
        )
    else:
        library_dir = dal_root / "lib" / "intel64"
        libraries = LINUX_HOST_LIBRARIES
    _create_libraries(library_dir, libraries)

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
            "SOABI": "cp312-win_amd64" if is_win else "cpython-312-x86_64-linux-gnu",
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
        onedal_major_binary_version=4,
        is_win=is_win,
        is_lin=not is_win,
        debug_build=debug_build,
    )

    build_type_args = [arg for arg in calls[0] if arg.startswith("-DCMAKE_BUILD_TYPE=")]
    assert build_type_args == [f"-DCMAKE_BUILD_TYPE={expected_build_type}"]
    assert f"-DSKLEARNEX_FREE_THREADING={'ON' if free_threading else 'OFF'}" in calls[0]
    assert "" not in calls[0]
    assert ("-GNinja" in calls[0]) is is_win
    assert not any(arg.startswith("-DSKLEARNEX_SANITIZER=") for arg in calls[0])
    assert calls[1][:2] == ["cmake", "--build"]
    assert Path(calls[1][2]).parts[-2:] == ("build", "backend_host")
    assert calls[1][3:] == ["-j", "1"]
    assert calls[2][:2] == ["cmake", "--build"]
    assert Path(calls[2][2]).parts[-2:] == ("build", "backend_host")
    assert calls[2][3:] == ["--target", "install"]

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
from pathlib import Path

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

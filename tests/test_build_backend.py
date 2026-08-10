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

"""Tests for the parts of the build driver CI cannot reach.

CI builds the release configuration on Linux, Windows and macOS, so the happy
path is covered there for real. What it never exercises is a debug build, a
oneDAL layout other than the one its own installation happens to use, or the
agreement between the two independent places that spell oneDAL library names.
Those are what this file pins down.
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
get_onedal_libraries = build_backend.get_onedal_libraries
get_onedal_library_filenames = build_backend.get_onedal_library_filenames

LINUX_HOST_LIBRARIES = (
    "libonedal.so.4",
    "libonedal_parameters.so.4",
    "libonedal_core.so.4",
    "libonedal_thread.so.4",
)
WINDOWS_HOST_LIBRARIES = (
    "onedal_dll.4.lib",
    "onedal_core_parameters_dll.4.lib",
    "onedal_core_dll.4.lib",
)


def _create_libraries(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).touch()


@pytest.mark.parametrize(
    "layout,is_win,libraries",
    [
        # Classic oneDAL install; also the layout CI itself uses.
        (("lib", "intel64"), False, LINUX_HOST_LIBRARIES),
        # Conda and some source installs flatten the arch directory away.
        (("lib",), False, LINUX_HOST_LIBRARIES),
        # Windows conda packages put libraries under the Library prefix.
        (("Library", "lib"), True, WINDOWS_HOST_LIBRARIES),
    ],
)
def test_get_onedal_library_dir_accepts_supported_layouts(
    tmp_path, layout, is_win, libraries
):
    library_dir = tmp_path.joinpath(*layout)
    _create_libraries(library_dir, libraries)

    assert _get_onedal_library_dir(
        tmp_path, "intel64", major_version=4, is_win=is_win
    ) == str(library_dir)


def test_get_onedal_library_dir_skips_incomplete_directory(tmp_path):
    # An arch directory left over from a different oneDAL major version must not
    # win over a complete one, which is what a plain isdir() check would do.
    flat = tmp_path / "lib"
    _create_libraries(flat, LINUX_HOST_LIBRARIES)
    _create_libraries(flat / "intel64", ("libonedal.so.3", "libonedal_core.so.3"))

    assert _get_onedal_library_dir(tmp_path, "intel64", major_version=4) == str(flat)


def test_get_onedal_library_dir_names_every_directory_and_miss(tmp_path):
    with pytest.raises(FileNotFoundError) as exc_info:
        _get_onedal_library_dir(tmp_path, "intel64", major_version=4)

    message = str(exc_info.value)
    assert str(tmp_path / "lib" / "intel64") in message
    assert str(tmp_path / "lib") in message
    assert "libonedal_thread.so.4" in message


@pytest.mark.parametrize("iface", ["daal", "host", "dpc", "spmd_dpc"])
@pytest.mark.parametrize(
    "is_win,is_mac,prefix,suffix",
    [
        (False, False, ":lib", ".so.4"),
        (False, True, "", ".4"),
        (True, False, "", ".4"),
    ],
)
def test_linker_names_and_filenames_describe_the_same_libraries(
    iface, is_win, is_mac, prefix, suffix
):
    # setup.py links by these names and the CMake backend looks for the files
    # they resolve to. A divergence between the two only shows up as a link
    # error inside CI, so pin the correspondence here.
    linker_names = get_onedal_libraries(
        iface, 4, is_win=is_win, is_mac=is_mac, use_parameters_lib=not is_win
    )
    filenames = get_onedal_library_filenames(
        iface, 4, is_win=is_win, is_mac=is_mac, use_parameters_lib=not is_win
    )
    assert len(linker_names) == len(filenames)

    for linker_name, filename in zip(linker_names, filenames):
        assert linker_name.startswith(prefix)
        assert linker_name.endswith(suffix)
        stem = linker_name[len(prefix) : -len(suffix)]
        if is_win:
            assert filename == f"{stem}.4.lib"
        elif is_mac:
            assert filename == f"lib{stem}.4.dylib"
        else:
            assert filename == f"lib{stem}.so.4"


def test_backend_libraries_extend_the_daal_library_set():
    # daal4py links the core libraries; the oneDAL backends add an interface
    # library on top. Nothing may drop a core library on the way.
    core = get_onedal_libraries("daal", 4)
    for iface, expected in (("host", "onedal"), ("dpc", "onedal_dpc")):
        libraries = get_onedal_libraries(iface, 4, use_parameters_lib=False)
        assert libraries[0] == f":lib{expected}.so.4"
        assert set(core).issubset(libraries)


def test_windows_library_names_match_cmake():
    # scripts/CMakeLists.txt spells the Windows import libraries out by hand,
    # including the onedal_core_parameters_*_dll names that do not follow the
    # pattern of the others. Compare against it directly.
    cmake_text = (REPO_ROOT / "scripts" / "CMakeLists.txt").read_text()
    in_cmake = set(re.findall(r'"(onedal[A-Za-z_]*_dll)\.\$\{ONEDAL_MAJOR', cmake_text))

    from_python = set()
    for iface in ("host", "dpc"):
        for name in get_onedal_libraries(iface, 4, is_win=True):
            from_python.add(name.rsplit(".", 1)[0])

    assert from_python == in_cmake


def test_unsupported_backend_is_rejected():
    with pytest.raises(ValueError, match="Unsupported oneDAL backend"):
        get_onedal_libraries("host_dpc", 4)


@pytest.mark.parametrize(
    "debug_build,expected_build_type", [(False, "Release"), (True, "Debug")]
)
def test_cmake_build_type_follows_the_debug_flag(
    monkeypatch, tmp_path, debug_build, expected_build_type
):
    # CI only ever builds the release configuration, so --debug silently
    # producing an optimized build would go unnoticed there.
    dal_root = tmp_path / "dal"
    _create_libraries(dal_root / "lib" / "intel64", LINUX_HOST_LIBRARIES)

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
            "LIBDEST": str(tmp_path / "python" / "lib" / "python"),
            "LIBDIR": str(tmp_path / "python" / "lib"),
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
        is_lin=True,
        debug_build=debug_build,
    )

    configure_args = calls[0]
    assert [arg for arg in configure_args if arg.startswith("-DCMAKE_BUILD_TYPE=")] == [
        f"-DCMAKE_BUILD_TYPE={expected_build_type}"
    ]
    # An empty argv entry is not portable across CMake versions and shows up as
    # a bare "" in the build log.
    assert "" not in configure_args

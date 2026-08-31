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
"""

import importlib.util
import re
from pathlib import Path

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

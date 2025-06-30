#! /usr/bin/env python
# ===============================================================================
# Copyright 2021 Intel Corporation
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

import re
import sys
from ctypes.util import find_library
from os.path import isfile
from os.path import join as jp


def find_defines(defines: list, file_obj):
    defines_dict = {define: "" for define in defines}
    for elem in file_obj:
        for define in defines:
            if f"#define {define}" in elem:
                match = re.match(rf"#define {define} (\d+)", elem)
                if match:
                    defines_dict[define] = match.group(1)
    return defines_dict


def get_onedal_version(dal_root, version_type="release"):
    """Parse oneDAL version strings"""

    if version_type not in ["release", "binary"]:
        raise ValueError(f'Incorrect version type "{version_type}"')

    header_candidates = [
        jp(dal_root, "include", "dal", "services", "library_version_info.h"),
        jp(dal_root, "Library", "include", "dal", "services", "library_version_info.h"),
        jp(dal_root, "include", "services", "library_version_info.h"),
    ]
    for candidate in header_candidates:
        if isfile(candidate):
            header_version = candidate
            break

    version = ""
    with open(header_version, "r") as header:
        if version_type == "release":
            version = find_defines(
                ["__INTEL_DAAL__", "__INTEL_DAAL_MINOR__", "__INTEL_DAAL_UPDATE__"],
                header,
            )
            version = (
                int(version["__INTEL_DAAL__"]) * 10000
                + int(version["__INTEL_DAAL_MINOR__"]) * 100
                + int(version["__INTEL_DAAL_UPDATE__"])
            )
        elif version_type == "binary":
            version = find_defines(
                ["__INTEL_DAAL_MAJOR_BINARY__", "__INTEL_DAAL_MINOR_BINARY__"], header
            )
            version = int(version["__INTEL_DAAL_MAJOR_BINARY__"]), int(
                version["__INTEL_DAAL_MINOR_BINARY__"]
            )
    return version


def find_library_custom_paths(alias, dal_root):
    if find_library(alias):
        return True
    IS_WIN = sys.platform in ["win32", "cygwin"]
    path_from_dal_root = (
        jp(dal_root, "Library", "bin") if IS_WIN else jp(dal_root, "lib", "intel64")
    )
    return alias in os.listdir(path_from_dal_root)


def get_onedal_shared_libs(dal_root):
    """Function to find which oneDAL shared libraries are available in the system"""
    lib_names = [
        "onedal",
        "onedal_core",
        "onedal_thread",
        "onedal_dpc",
        "onedal_parameters",
    ]
    major_bin_version, _ = get_onedal_version(dal_root, "binary")
    found_libraries = []
    for lib_name in lib_names:
        possible_aliases = [
            lib_name,
            f"lib{lib_name}.so.{major_bin_version}",
            f"lib{lib_name}.{major_bin_version}.dylib",
            f"{lib_name}.{major_bin_version}.dll",
        ]
        if any(find_library_custom_paths(alias, dal_root) for alias in possible_aliases):
            found_libraries.append(lib_name)
    return found_libraries

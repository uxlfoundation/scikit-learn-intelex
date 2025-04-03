#! /usr/bin/env python
# ==============================================================================
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
# ==============================================================================

import argparse
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--channels", nargs="+", default=["pypi"])
args = parser.parse_args()

CHANNELS = args.channels
# image versions are pinned to exact number instead of "latest"
# to avoid unexpected failures when images are updated
ACTIVATE = {
    "ubuntu": "conda activate",
    "windows": "call activate",
}

print(CHANNELS)


def generate_python_versions(file="README.md"):
    """Attempt to centralize the supported versions in the right location: the README.
    Take it from a badge which lists the supported versions."""
    regex = (
        r"(?<=\[python version\]\(https://img.shields.io/badge/python-).*(?=-blue\)\])"
    )
    sep = "%20%7C%20"
    pydefaults = ["3.9", "3.10", "3.11"]
    if os.path.isfile(file):
        with open(file, "r") as f:
            pydefaults = re.findall(regex, f.read())[0].split(sep)
    return pydefaults


PYTHON_VERSIONS = generate_python_versions()


def collect_azp_CI_OS_images(file=f".ci{os.sep}pipeline{os.sep}ci.yml"):
    """Attempt to centralize the supported version from the azp CI pipeline, which
    represents the currently tested versions in Azure Pipelines."""
    regex = r"(?<=vmImage: ').*(?=')"
    sysdefaults = ["ubuntu-22.04", "windows-2022"]
    if os.path.isfile(file):
        with open(file, "r") as f:
            # find unique values with set
            sysdefaults = list(set(re.findall(regex, f.read())))
    return sysdefaults


OS_VERSIONS = collect_azp_CI_OS_images()


res_enum = {}
for channel in CHANNELS:
    for python_version in PYTHON_VERSIONS:
        for vmOS in OS_VERSIONS:
            res_key = channel + " - " + "python" + python_version + " - " + vmOS
            res_enum[res_key] = {}
            res_enum[res_key]["python.version"] = python_version
            res_enum[res_key]["imageName"] = vmOS
            # collect only the OS name, not version via split
            res_enum[res_key]["conda.activate"] = ACTIVATE[vmOS.split("-")[0]]
            res_enum[res_key]["conda.channel"] = channel

sys.stderr.write("##vso[task.setVariable variable=legs;isOutput=true]{}".format(res_enum))

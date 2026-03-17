#!/usr/bin/env python
# ==============================================================================
# Copyright 2020 Intel Corporation
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

# coding: utf-8
import argparse
import os.path
import sys
import warnings

import sklearn

try:
    from packaging.version import Version
except ImportError:
    from distutils.version import LooseVersion as Version

from sklearn import __version__ as sklearn_version
from yaml import FullLoader
from yaml import load as yaml_load


def evaluate_cond(cond, v):
    if cond.startswith(">="):
        return Version(v) >= Version(cond[2:])
    if cond.startswith("<="):
        return Version(v) <= Version(cond[2:])
    if cond.startswith("=="):
        return Version(v) == Version(cond[2:])
    if cond.startswith("!="):
        return Version(v) != Version(cond[2:])
    if cond.startswith("<"):
        return Version(v) < Version(cond[1:])
    if cond.startswith(">"):
        return Version(v) > Version(cond[1:])
    warnings.warn(
        'Test selection condition "{0}" should start with '
        ">=, <=, ==, !=, < or > to compare to version of scikit-learn run. "
        "The test will not be deselected".format(cond)
    )
    return False


def filter_by_version_and_platform(entry, sk_ver):
    if not entry:
        return None
    t = entry.split(" ")
    if len(t) == 1:
        return entry
    elif len(t) == 2:
        t.append(None)
    if len(t) != 3:
        return None
    test_name, cond, platform = t
    if platform is not None and platform != sys.platform:
        return None
    conds = cond.split(",")
    if all([evaluate_cond(cond, sk_ver) for cond in conds]):
        return test_name
    return None


def create_pytest_switches(
    filename, absolute, reduced, public, gpu, preview, base_dir=None
):
    pytest_switches = []
    if os.path.exists(filename):
        with open(filename, "r") as fh:
            dt = yaml_load(fh, Loader=FullLoader)

        sklearn_dir = os.path.dirname(sklearn.__file__)

        if absolute:
            base_dir = os.path.relpath(sklearn_dir, os.path.expanduser("~")) + "/"
        elif base_dir is None:
            base_dir = ""
        elif not base_dir.endswith("/"):
            base_dir += "/"

        filtered_deselection = [
            filter_by_version_and_platform(test_name, sklearn_version)
            for test_name in dt.get("deselected_tests", [])
        ]
        if reduced:
            filtered_deselection.extend(
                [
                    filter_by_version_and_platform(test_name, sklearn_version)
                    for test_name in dt.get("reduced_tests", [])
                ]
            )
        if public:
            filtered_deselection.extend(
                [
                    filter_by_version_and_platform(test_name, sklearn_version)
                    for test_name in dt.get("public", [])
                ]
            )
        if gpu:
            filtered_deselection.extend(
                [
                    filter_by_version_and_platform(test_name, sklearn_version)
                    for test_name in dt.get("gpu", [])
                ]
            )
        if preview:
            filtered_deselection.extend(
                [
                    filter_by_version_and_platform(test_name, sklearn_version)
                    for test_name in dt.get("preview", [])
                ]
            )
        pytest_switches = []
        for test_name in filtered_deselection:
            if test_name:
                pytest_switches.extend(["--deselect=" + base_dir + test_name])

        # Files that crash at collection time need --ignore instead of --deselect
        for entry in dt.get("ignored_tests", []):
            rel_path = filter_by_version_and_platform(entry, sklearn_version)
            if rel_path:
                pytest_switches.append(
                    "--ignore=" + os.path.join(sklearn_dir, *rel_path.split("/"))
                )
    return pytest_switches


if __name__ == "__main__":
    argParser = argparse.ArgumentParser(
        prog="deselect_tests.py",
        description="Produce pytest CLI options to deselect tests specified in yaml file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argParser.add_argument("conf_file", nargs=1, type=str)
    argParser.add_argument("--absolute", action="store_true")
    argParser.add_argument("--reduced", action="store_true")
    argParser.add_argument("--public", action="store_true")
    argParser.add_argument("--gpu", action="store_true")
    argParser.add_argument("--preview", action="store_true")
    argParser.add_argument("--base-dir", type=str, default=None)
    args = argParser.parse_args()

    fn = args.conf_file[0]
    if os.path.exists(fn):
        print(
            " ".join(
                create_pytest_switches(
                    fn,
                    args.absolute,
                    args.reduced,
                    args.public,
                    args.gpu,
                    args.preview,
                    args.base_dir,
                )
            )
        )

# ===============================================================================
# Copyright 2023 Intel Corporation
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

"""Script for running the Scikit-learn pytest test suite using sklearnex patching.

Notes
-----
The script will pass all additional arguments to pytest by default. The options
listed below dependent on environment variables can be manually set instead via
the command line using this functionality.

It adds the `-d` and `--device` flags which will use sklearnex's target_offload
functionality in order to run the Scikit-learn test suite on a selected device.
Supported options are limited to 'cpu' and 'gpu' which default to the first
option available of that type via the SYCL device_selector.

The script reads the JSON_REPORT_FILE environment variable as the output location
for the pytest json report plugin when set.

Run_sklearn_test.py also enables coverage statistics for the onedal and sklearnex
modules to the file listed in the COVERAGE_RCFILE environment variable.

It will acquire requisite deselections and selections of tests from the
DESELECTED_TESTS and SELECTED_TESTS environment variables respectively.

The environment variable SCIPY_ARRAY_API is set by default, and impacts the
operation of the Scikit-learn test suite.
"""

from sklearnex import patch_sklearn

patch_sklearn()

import argparse
import os
import sys

import pytest
import sklearn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="none",
        help="device name",
        choices=["none", "cpu", "gpu"],
    )
    args, pytest_args = parser.parse_known_args()

    sklearn_file_dir = os.path.dirname(sklearn.__file__)
    os.chdir(sklearn_file_dir)

    if os.environ["SELECTED_TESTS"] == "all":
        os.environ["SELECTED_TESTS"] = ""

    os.environ["SCIPY_ARRAY_API"] = "1"

    pytest_args += [
        f"--rootdir={sklearn_file_dir} "
        f'{os.environ["DESELECTED_TESTS"]} {os.environ["SELECTED_TESTS"]}'.split(" ")
    ]

    if rc := os.getenv("COVERAGE_RCFILE"):
        pytest_args += [
            "--cov=onedal",
            "--cov=sklearnex",
            "--cov-branch",
            f"--cov-config={rc}",
            "--cov-report=",
        ]

    if json_file := os.getenv("JSON_REPORT_FILE"):
        pytest_args += ["--json-report", f"--json-report-file={json_file}"]

    while "" in pytest_args:
        pytest_args.remove("")

    if args.device != "none":
        with sklearn.config_context(target_offload=args.device):
            return_code = pytest.main(pytest_args)
    else:
        return_code = pytest.main(pytest_args)

    sys.exit(int(return_code))

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

import os

os.environ["SCIPY_ARRAY_API"] = "1"
from sklearnex import patch_sklearn

patch_sklearn()

import argparse
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
    args, extra_args = parser.parse_known_args()

    sklearn_file_dir = os.path.dirname(sklearn.__file__)
    os.chdir(sklearn_file_dir)

    if os.environ["SELECTED_TESTS"] == "all":
        os.environ["SELECTED_TESTS"] = ""

    pytest_args = (
        f"--rootdir={sklearn_file_dir} "
        f'{os.environ["DESELECTED_TESTS"]} {os.environ["SELECTED_TESTS"]}'.split(" ")
    )

    if rc := os.getenv("COVERAGE_RCFILE"):
        pytest_args += (
            "--cov=onedal",
            "--cov=sklearnex",
            "--cov-branch",
            f"--cov-config={rc}",
            "--cov-report=",
        )
    if json_file := os.getenv("JSON_REPORT_FILE"):
        pytest_args += ["--json-report", f"--json-report-file={json_file}"]

    while "" in pytest_args:
        pytest_args.remove("")

    if extra_args:
        pytest_args += extra_args

    if args.device != "none":
        with sklearn.config_context(target_offload=args.device):
            return_code = pytest.main(pytest_args)
    else:
        return_code = pytest.main(pytest_args)

    sys.exit(int(return_code))

#! /usr/bin/env python
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
import argparse
import sys

from daal4py.sklearn._utils import sklearn_check_version


def parse_tests_tree(entry, prefix=""):
    global tests_list

    if isinstance(entry, dict):
        for key, value in entry.items():
            parse_tests_tree(value, f"{prefix}/{key}" if prefix != "" else key)
    elif isinstance(entry, list):
        for value in entry:
            parse_tests_tree(value, prefix)
    elif isinstance(entry, str):
        tests_list.append(f"{prefix}/{entry}" if prefix != "" else entry)
    else:
        raise ValueError(f"Unknown type {type(entry)} in tests map")


tests_map = {
    #"cluster/tests": ["test_dbscan.py", "test_k_means.py"],
    #"covariance/tests": "test_covariance.py",
    #"decomposition/tests": "test_pca.py",
    #"ensemble/tests": "test_forest.py",
    #"linear_model/tests": ["test_base.py", "test_coordinate_descent.py", "test_ridge.py"],
    #"manifold/tests": "test_t_sne.py",
    "model_selection/tests": "test_validation.py",
    #"neighbors/tests": ["test_lof.py", "test_neighbors.py", "test_neighbors_pipeline.py"],
    "svm/tests": ["test_sparse.py", "test_svm.py"],
}
if sklearn_check_version("1.2"):
    tests_map["tests"] = ["test_public_functions.py"]


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--base-dir", type=str, default="")
    args = arg_parser.parse_args()

    tests_list = []
    parse_tests_tree(tests_map, args.base_dir)
    result = ""
    for test in tests_list:
        result += test + " "
    # correct paths for non-Unix envs
    if sys.platform in ["win32", "cygwin"]:
        result = result.replace("/", "\\")
    print(result[:-1])

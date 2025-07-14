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

# Note: see file 'scikit-learn-tests.md' for instructions about usage
# of this script:
# https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/scikit-learn-tests.md
import json
import sys
from typing import Any


def load_json_from_file(fname: str) -> dict:
    with open(fname, "r") as f:
        return json.load(f)


def dct_wo_key(dct: dict[str, Any], k_remove: str) -> dict[str, Any]:
    return {k: v for k, v in dct.items() if k != k_remove}


def index_dict_by_key(dct: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    return {elt[key]: dct_wo_key(elt, key) for elt in dct}


def produce_differences(
    fname1: str, fname2: str, pretty_name1: str, pretty_name2: str
) -> dict:
    json1 = load_json_from_file(fname1)
    json2 = load_json_from_file(fname2)
    results1 = index_dict_by_key(json1["tests"], "nodeid")
    results2 = index_dict_by_key(json2["tests"], "nodeid")
    tests_common: set[str] = results1.keys() & results2.keys()
    dct_differences: dict[str, dict] = dict()
    for test_id_key in tests_common:
        test_result1 = results1[test_id_key]
        test_result2 = results2[test_id_key]

        if test_result1["outcome"] != test_result2["outcome"]:
            dct_differences[test_id_key] = {
                pretty_name1: test_result1,
                pretty_name2: test_result2,
            }
    return dct_differences


def save_dct_to_json(dct: dict, out_fname: str) -> None:
    with open(out_fname, "w") as f:
        json.dump(
            dct,
            f,
            indent="\t",
            separators=(", ", ": "),
            sort_keys=True,
        )


def read_arguments() -> tuple[str, str, str, str, str]:
    def get_cmd_arg(arg_prefix: str) -> str:
        sel_arg = None
        for passed_arg in sys.argv:
            if passed_arg.startswith(arg_prefix):
                sel_arg = passed_arg
        if sel_arg is None:
            raise ValueError(f"Must supply argument '{sel_arg}'.")
        sys.argv = [elt for elt in sys.argv if elt != sel_arg]
        return sel_arg.split("=", maxsplit=1)[1]

    fname1 = get_cmd_arg("--json1")
    fname2 = get_cmd_arg("--json2")
    pretty_name1 = get_cmd_arg("--name1")
    pretty_name2 = get_cmd_arg("--name2")
    out_fname = get_cmd_arg("--output")
    return fname1, fname2, pretty_name1, pretty_name2, out_fname


if __name__ == "__main__":
    fname1, fname2, pretty_name1, pretty_name2, out_fname = read_arguments()
    diffs_dct = produce_differences(fname1, fname2, pretty_name1, pretty_name2)
    save_dct_to_json(diffs_dct, out_fname)

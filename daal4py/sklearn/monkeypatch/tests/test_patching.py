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

import os
import pathlib
import re
import subprocess
import sys

import pytest
from _models_info import TO_SKIP


def get_branch(s):
    if len(s) == 0:
        return "NO INFO"
    for i in s:
        if "failed to run accelerated version, fallback to original Scikit-learn" in i:
            return "was in OPT, but go in Scikit"
    for i in s:
        if "running accelerated version" in i:
            return "OPT"
    return "Scikit"


def run_parse(mas, result):
    print("\n[run_parse] mas:")
    for idx, val in enumerate(mas):
        print(f"  mas[{idx}] = {val!r}")

    try:
        name, dtype = mas[0].split()
        print(f"[run_parse] Parsed name = {name}, dtype = {dtype}")
    except ValueError as e:
        print(f"[run_parse] ERROR while splitting mas[0]: {mas[0]!r}")
        print(f"[run_parse] .split() gives: {mas[0].split()}")
        raise

    temp = []
    INFO_POS = 6
    for i in range(1, len(mas)):
        print(f"[run_parse] Processing mas[{i}]: {mas[i]!r}")
        mas[i] = mas[i][INFO_POS:]
        print(f"[run_parse] After trimming INFO: {mas[i]!r}")
        if not mas[i].startswith("sklearn"):
            ind = name + " " + dtype + " " + mas[i]
            branch = get_branch(temp)
            print(f"[run_parse] Adding result: {ind!r} => {branch}")
            result[ind] = branch
            temp.clear()
        else:
            temp.append(mas[i])
            print(f"[run_parse] Appended to temp: {temp!r}")


def get_result_log():
    os.environ["IDP_SKLEARN_VERBOSE"] = "INFO"
    absolute_path = str(pathlib.Path(__file__).parent.absolute())
    try:
        process = subprocess.check_output(
            [
                sys.executable,
                os.sep.join([absolute_path, "utils", "_launch_algorithms.py"]),
            ]
        )
    except subprocess.CalledProcessError as e:
        print("[get_result_log] Subprocess failed:")
        print(e)
        exit(1)

    mas = []
    result = {}
    decoded = process.decode().split("\n")
    print("[get_result_log] Process output:")
    for line in decoded:
        print(f"  {line!r}")

    for i in decoded:
        if not i.startswith("INFO") and len(mas) != 0:
            print(f"[get_result_log] Switching block at line: {i!r}")
            run_parse(mas, result)
            mas.clear()
            mas.append(i.strip())
        else:
            mas.append(i.strip())
    del os.environ["IDP_SKLEARN_VERBOSE"]
    return result


result_log = get_result_log()


@pytest.mark.parametrize("configuration", result_log)
def test_patching(configuration):
    if "OPT" in result_log[configuration]:
        return
    for skip in TO_SKIP:
        if re.search(skip, configuration) is not None:
            pytest.skip("SKIPPED", allow_module_level=False)
    raise ValueError("Test patching failed: " + configuration)

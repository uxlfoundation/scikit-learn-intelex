<!--
******************************************************************************
* Copyright contributors to the oneDAL project
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# Running the scikit-learn test suite

The Extension for scikit-learn* is regularly tested for correctness through the test suite of scikit-learn* itself executed with patching applied, referred throughout the CI jobs and files as 'conformance testing'.

To execute the scikit-learn* conformance tests, the following script can be used:

```shell
./.ci/scripts/run_sklearn_tests.sh
```

Note that some tests are known to produce failures - for example, scikit-learn's [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) allows an argument `copy_X`, and one of their tests checks that passing `copy_X=False` modifies the 'X' input in-place, while the extension never modifies this data regardless of the argument `copy_X`, and hence the test would show a failure under a patched scikit-learn*, even though the results do not change.

Cases that are known to fail are not executed during these conformance test. The list of deselected tests can be found under [deselected_tests.yaml](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/deselected_tests.yaml).

Individual tests can be executed through the underlying `.py` file that the `.sh` script executes, and other custom selections or deselections can be changed on-the-fly there through usage of environment variables - for example:

```shell
SELECTED_TESTS=all DESELECTED_TESTS="" python .ci/scripts/run_sklearn_tests.py
```

_**Note:** If building the extension modules in-place [per the instructions here](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md#build-intelr-extension-for-scikit-learn), it requires also setting `$PYTHONPATH` for this script._

The tests can also be made to run on GPU, either by passing argument `gpu` to `run_sklearn_tests.sh`, or by passing argument `--device <device name>` to  `run_sklearn_tests.py` - example:
```shell
./.ci/scripts/run_sklearn_tests.sh gpu
```

Note that functionalities under [preview](https://uxlfoundation.github.io/scikit-learn-intelex/latest/preview.html) are not tested by default - in order to test them, it's necessary to set environment variable `SKLEARNEX_PREVIEW=1` to enable patching of such functionalities before executing either of these scripts (`.sh` / `.py`). The `.sh` script by default will take care of deselecting tests that involve preview functionalities when this environment variable is not set.

## Producing a test report

Optionally, a JSON report of the results can be produced (requires package `pytest-json-report`) by setting an environment variable `JSON_REPORT_FILE`, indicating the location where to produce a JSON output file - note that the test runner changes the PyTest root directory, so it should be specified as an absolute path, or otherwise will get written into the `site-packages` folder for `sklearn`:

```shell
SELECTED_TESTS=all \
DESELECTED_TESTS="" \
JSON_REPORT_FILE="$(pwd)/sklearn_test_results.json" \
    python .ci/scripts/run_sklearn_tests.py
```

## Comparing test reports

A small utility to compare two JSON test reports is provided under [tests/util_compare_json_reports.py](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/tests/util_compare_json_reports.py), which can be useful for example when comparing changes before and after a given commit.

The file is a python script which produces a new JSON output file highlighting the tests that had different outcomes between two JSON reports. It needs to be executed with the following arguments, prefixed with two dashes and with the value passed after an equal sign (e.g. `--arg1=value`):

* `json1`: path to a first JSON report file from `pytest-json-report`.
* `json2`: path to a second JSON report file from `pytest-json-report`.
* `name1`: name that the tests from the first file will use as JSON keys in the generated output file.
* `name2`: name that the tests from the second file will use as JSON keys in the generated output file.
* `output`: file name where to save the result JSON file that highlights the differences.

Example:
```shell
python tests/util_compare_json_reports.py \
    --json1=logs_before.json \
    --json2=logs_after.json \
    --name1="before" \
    --name2="after" \
    --output="diffs_before_after.json"
```

The result will be a new JSON file which will contain only entries for tests that were present in both files and which had different outcomes, with a structure as follows:
```
"test_name": { # taken from 'nodeid' in the pytest json reports
    <name1>: { # taken from argument 'name1'
        ...    # json from entry in pytest report under 'tests', minus key 'nodeid'
    },
    <name2>: { # taken from argument 'name2'
        ...    # json from entry in pytest report under 'tests', minus key 'nodeid'
    }
}
```

# Running the scikit-learn test suite

The Extension for scikit-learn* is regularly tested for correctness through the test suite of scikit-learn* itself executed with patching applied, referred throughout the CI jobs and files as 'conformance testing'.

To execute the scikit-learn* conformance tests, the following script can be used:

```shell
./.ci/scripts/run_sklearn_tests.sh
```

Note that some tests are known to produce failures - for example, scikit-learn's [LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) allows an argument `copy_X`, and one of their tests checks that passing `copy_X=False` modifies the 'X' input in-place, while the extension never modifies this data regardless of the argument `copy_X`, and hence the test would show a failure under a patched scikit-learn*, even though the results do not change.

Cases that are known to fail are not executed during these conformance test. The list of deselected tests can be found under [deselected_tests.yaml](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/deselected_tests.yaml).

Individual tests can be executed through the underlying .py file that the .sh script executes, and other custom selections or deselections can be changed on-the-fly there through usage of environment variables - for example:

```shell
SELECTED_TESTS=all DESELECTED_TESTS="" python .ci/scripts/run_sklearn_tests.py
```

_**Note:** If building the extension modules in-place [per the instructions here](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md#build-intelr-extension-for-scikit-learn), it requires also setting `$PYTHONPATH` for this script._

Optionally, a JSON report of the results can be produced (requires package `pytest-json-report`) by setting an environment variable `JSON_REPORT_FILE`, indicating the location where to produce a JSON output file - note that the test runner changes the PyTest root directory, so it should be specified as an absolute path, or otherwise will get written into the `site-packages` folder for `sklearn`:

```shell
SELECTED_TESTS=all DESELECTED_TESTS="" JSON_REPORT_FILE="$(pwd)/sklearn_test_results.json" python .ci/scripts/run_sklearn_tests.py
```

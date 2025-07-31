# ==============================================================================
# Copyright 2024 Intel Corporation
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

import importlib.util
import io
import os
import pathlib
import pkgutil
import re
import subprocess
import sys
import trace
from contextlib import redirect_stdout
from multiprocessing import Pipe, Process, get_context

import pytest
from sklearn.base import BaseEstimator
from sklearn.utils import all_estimators

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.test_common import _check_primitive_usage_ban
from onedal.tests.utils._dataframes_support import test_frameworks
from sklearnex.base import oneDALEstimator
from sklearnex.tests.utils import (
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    UNPATCHED_MODELS,
    call_method,
    gen_dataset,
    gen_models_info,
)

TARGET_OFFLOAD_ALLOWED_LOCATIONS = [
    "_config.py",
    "_device_offload.py",
    "test",
    "svc.py",
    "svm" + os.sep + "_common.py",
]

_DESIGN_RULE_VIOLATIONS = {
    "PCA-fit_transform-call_validate_data": "calls both 'fit' and 'transform'",
    "IncrementalEmpiricalCovariance-score-call_validate_data": "must call clone of itself",
    "SVC(probability=True)-fit-call_validate_data": "SVC fit can use sklearn estimator",
    "NuSVC(probability=True)-fit-call_validate_data": "NuSVC fit can use sklearn estimator",
    "LogisticRegression-score-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression-fit-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression-predict-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression-predict_log_proba-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression-predict_proba-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "KNeighborsClassifier-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-score-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-predict-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-predict_proba-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-score-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-predict-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-radius_neighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors-radius_neighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-score-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-predict-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-predict_proba-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsClassifier(algorithm='brute')-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-score-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-predict-n_jobs_check": "uses daal4py for cpu in onedal",
    "KNeighborsRegressor(algorithm='brute')-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-radius_neighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "NearestNeighbors(algorithm='brute')-radius_neighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor(novelty=True)-fit-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor(novelty=True)-kneighbors-n_jobs_check": "uses daal4py for cpu in onedal",
    "LocalOutlierFactor(novelty=True)-kneighbors_graph-n_jobs_check": "uses daal4py for cpu in onedal",
    "LogisticRegression(solver='newton-cg')-score-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression(solver='newton-cg')-fit-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression(solver='newton-cg')-predict-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression(solver='newton-cg')-predict_log_proba-n_jobs_check": "uses daal4py for cpu in sklearnex",
    "LogisticRegression(solver='newton-cg')-predict_proba-n_jobs_check": "uses daal4py for cpu in sklearnex",
}


def test_target_offload_ban():
    """This test blocks the use of target_offload in
    in sklearnex files. Offloading computation to devices
    via target_offload should only occur externally, and not
    within the architecture of the sklearnex classes. This
    is for clarity, traceability and maintainability.
    """
    output = _check_primitive_usage_ban(
        primitive_name="target_offload",
        package="sklearnex",
        allowed_locations=TARGET_OFFLOAD_ALLOWED_LOCATIONS,
    )
    output = "\n".join(output)
    assert output == "", f"target offloading is occurring in: \n{output}"


def _sklearnex_walk(func):
    """this replaces checks on pkgutils to look through sklearnex
    folders specifically"""

    def wrap(*args, **kwargs):
        if "prefix" in kwargs and kwargs["prefix"] == "sklearn.":
            kwargs["prefix"] = "sklearnex."
        if "path" in kwargs:
            # force root to sklearnex
            kwargs["path"] = [str(pathlib.Path(__file__).parent.parent)]
        for pkginfo in func(*args, **kwargs):
            # Do not allow spmd to be yielded
            if "spmd" not in pkginfo.name.split("."):
                yield pkginfo

    return wrap


def test_class_trailing_underscore_ban(monkeypatch):
    """Trailing underscores are defined for sklearn to be signatures of a fitted
    estimator instance, sklearnex extends this to the classes as well"""
    monkeypatch.setattr(pkgutil, "walk_packages", _sklearnex_walk(pkgutil.walk_packages))
    estimators = all_estimators()  # list of tuples
    for name, obj in estimators:
        if "preview" not in obj.__module__ and "daal4py" not in obj.__module__:
            # properties also occur in sklearn, especially in deprecations and are expected
            # to error if queried and the estimator is not fitted
            assert all(
                [
                    isinstance(getattr(obj, attr), property)
                    or (attr.startswith("_") or not attr.endswith("_"))
                    for attr in dir(obj)
                ]
            ), f"{name} contains class attributes which have a trailing underscore but no leading one"


def test_all_estimators_covered(monkeypatch):
    """Check that all estimators defined in sklearnex are available in either the
    patch map or covered in special testing via SPECIAL_INSTANCES. The estimator
    must inherit sklearn's BaseEstimator and must not have a leading underscore.
    The sklearnex.spmd and sklearnex.preview packages are not tested.
    """
    monkeypatch.setattr(pkgutil, "walk_packages", _sklearnex_walk(pkgutil.walk_packages))
    estimators = all_estimators()  # list of tuples
    uncovered_estimators = []
    for name, obj in estimators:
        # do nothing if defined in preview
        if "preview" not in obj.__module__ and not (
            any([issubclass(est, obj) for est in PATCHED_MODELS.values()])
            or any([issubclass(est.__class__, obj) for est in SPECIAL_INSTANCES.values()])
        ):
            uncovered_estimators += [".".join([obj.__module__, name])]

    assert (
        uncovered_estimators == []
    ), f"{uncovered_estimators} are currently not included"


def test_oneDALEstimator_inheritance(monkeypatch):
    """All sklearnex estimators should inherit the oneDALEstimator class, sklearnex-only
    estimators should have it inherit oneDAL estimator one step before BaseEstimator in the
    mro.  This is only strictly set for non-preview estimators"""
    monkeypatch.setattr(pkgutil, "walk_packages", _sklearnex_walk(pkgutil.walk_packages))
    estimators = all_estimators()  # list of tuples
    for name, obj in estimators:
        if "preview" not in obj.__module__ and "daal4py" not in obj.__module__:
            assert issubclass(
                obj, oneDALEstimator
            ), f"{name} does not inherit the oneDALEstimator"
            # oneDAL estimator should be inherited from before BaseEstimator
            mro = obj.__mro__
            assert mro.index(oneDALEstimator) < mro.index(
                BaseEstimator
            ), f"incorrect mro in {name}"
            if not any([issubclass(obj, est) for est in UNPATCHED_MODELS.values()]):
                assert (
                    mro[mro.index(oneDALEstimator) + 1] is BaseEstimator
                ), f"oneDALEstimator should be inherited just before BaseEstimator in {name}"


def test_frameworks_lazy_import(monkeypatch):
    """Check that all estimators defined in sklearnex do not actively
    load data frameworks which are not numpy or pandas.
    """
    active = ["numpy", "pandas", "dpctl.tensor"]
    # handle naming conventions for data frameworks in testing
    frameworks = test_frameworks.replace("dpctl", "dpctl.tensor")
    frameworks = frameworks.replace("array_api", "array_api_strict")
    lazy = ",".join([i for i in frameworks.split(",") if i not in active])
    if not lazy:
        pytest.skip("No lazily-imported data frameworks available in testing")

    monkeypatch.setattr(pkgutil, "walk_packages", _sklearnex_walk(pkgutil.walk_packages))
    estimators = all_estimators()  # list of tuples

    filtered_modules = []
    for name, obj in estimators:
        # do not test spmd or preview, as they are exempt
        if "preview" not in obj.__module__ and "spmd" not in obj.__module__:
            filtered_modules += [obj.__module__]

    modules = ",".join(filtered_modules)

    # import all modules with estimators and check sys.modules for the lazily-imported data
    # frameworks. It is done in a subprocess to isolate the impact of testing infrastructure
    # on sys.modules, which may have actively loaded those frameworks into the test env
    teststr = (
        "import sys,{mod};assert all([i not in sys.modules for i in '{l}'.split(',')])"
    )
    cmd = [sys.executable, "-c", teststr.format(mod=modules, l=lazy)]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise AssertionError(f"a framework in '{lazy}' is being actively loaded") from e


def _fullpath(path):
    return os.path.realpath(os.path.expanduser(path))


_TRACE_ALLOW_DICT = {
    i: _fullpath(os.path.dirname(importlib.util.find_spec(i).origin))
    for i in ["sklearn", "sklearnex", "onedal", "daal4py"]
}


def _whitelist_to_blacklist():
    """block all standard library, built-in or site packages which are not
    related to sklearn, daal4py, onedal or sklearnex"""

    def _commonpath(inp):
        # ValueError generated by os.path.commonpath when it is on a separate drive
        try:
            return os.path.commonpath(inp)
        except ValueError:
            return ""

    blacklist = []
    for path in sys.path:
        fpath = _fullpath(path)
        try:
            # if candidate path is a parent directory to any directory in the whitelist
            if any(
                [_commonpath([i, fpath]) == fpath for i in _TRACE_ALLOW_DICT.values()]
            ):
                # find all sub-paths which are not in the whitelist and block them
                # they should not have a common path that is either the whitelist path
                # or the sub-path (meaning one is a parent directory of the either)
                for f in os.scandir(fpath):
                    temppath = _fullpath(f.path)
                    if all(
                        [
                            _commonpath([i, temppath]) not in [i, temppath]
                            for i in _TRACE_ALLOW_DICT.values()
                        ]
                    ):
                        blacklist += [temppath]
            # add path to blacklist if not a sub path of anything in the whitelist
            elif all([_commonpath([i, fpath]) != i for i in _TRACE_ALLOW_DICT.values()]):
                blacklist += [fpath]
        except FileNotFoundError:
            blacklist += [fpath]
    return blacklist


_TRACE_BLOCK_LIST = _whitelist_to_blacklist()


def sklearnex_trace(estimator_name, method_name):
    """Generate a trace of all function calls in calling estimator.method.

    Parameters
    ----------
    estimator_name : str
        name of estimator which is a key from PATCHED_MODELS or SPECIAL_INSTANCES

    method_name : str
        name of estimator method which is to be traced and stored

    Returns
    -------
    text: str
        Returns a string output (captured stdout of a python Trace call). It is a
        modified version to be more informative, completed by a monkeypatching
        of trace._modname.
    """
    # get estimator
    est = (
        PATCHED_MODELS[estimator_name]()
        if estimator_name in PATCHED_MODELS
        else SPECIAL_INSTANCES[estimator_name]
    )

    # get dataset
    X, y = gen_dataset(est)[0]
    # fit dataset if method does not contain 'fit'
    if "fit" not in method_name:
        est.fit(X, y)

    # monkeypatch new modname for clearer info
    orig_modname = trace._modname
    try:
        # initialize tracer to have a more verbose module naming
        # this impacts ignoremods, but it is not used.
        trace._modname = _fullpath
        tracer = trace.Trace(
            count=0,
            trace=1,
            ignoredirs=_TRACE_BLOCK_LIST,
        )
        # call trace on method with dataset
        f = io.StringIO()
        with redirect_stdout(f):
            tracer.runfunc(call_method, est, method_name, X, y)
        return f.getvalue()
    finally:
        trace._modname = orig_modname


def _trace_daemon(pipe):
    """function interface for the other process. Information
    exchanged using a multiprocess.Pipe"""
    # a sent value with inherent conversion to False will break
    # the while loop and complete the function
    while key := pipe.recv():
        try:
            text = sklearnex_trace(*key)
        except:
            # catch all exceptions and pass back,
            # this way the process still runs
            text = ""
        finally:
            pipe.send(text)


class _FakePipe:
    """Minimalistic representation of a multiprocessing.Pipe for test development.
    This allows for running sklearnex_trace in the parent process"""

    _text = ""

    def send(self, key):
        self._text = sklearnex_trace(*key)

    def recv(self):
        return self._text


@pytest.fixture(scope="module")
def isolated_trace():
    """Generates a separate python process for isolated sklearnex traces.

    It is a module scope fixture due to the overhead of importing all the
    various dependencies and is done once before all the various tests.
    Each test will first check a cached value, if not existent it will have
    the waiting child process generate the trace and return the text for
    caching on its behalf. The isolated process is stopped at test teardown.

    Yields
    -------
    pipe_parent: multiprocessing.Connection
        one end of a duplex pipe to be used by other pytest fixtures for
        communicating with the special isolated tracing python instance
        for sklearnex estimators.
    """
    # yield _FakePipe()
    try:
        # force use of 'spawn' to guarantee a clean python environment
        # from possible coverage arc tracing
        ctx = get_context("spawn")
        pipe_parent, pipe_child = ctx.Pipe()
        p = ctx.Process(target=_trace_daemon, args=(pipe_child,), daemon=True)
        p.start()
        yield pipe_parent
    finally:
        # guarantee closing of the process via a try-catch-finally
        # passing False terminates _trace_daemon's loop
        pipe_parent.send(False)
        pipe_parent.close()
        pipe_child.close()
        p.join()
        p.close()


@pytest.fixture
def estimator_trace(estimator, method, cache, isolated_trace):
    """Create cache of all function calls in calling estimator.method.

    Parameters
    ----------
    estimator : str
        name of estimator which is a key from PATCHED_MODELS or SPECIAL_INSTANCES

    method : str
        name of estimator method which is to be traced and stored

    cache: pytest.fixture (standard)

    isolated_trace: pytest.fixture (test_common.py)

    Returns
    -------
    dict: [calledfuncs, tracetext, modules, callinglines]
        Returns a list of important attributes of the trace.
        calledfuncs is the list of called functions, tracetext is the
        total text output of the trace as a string, modules are the
        module locations  of the called functions (must be from daal4py,
        onedal, sklearn, or sklearnex), and callinglines is the line
        which calls the function in calledfuncs
    """
    key = "-".join((str(estimator), method))
    flag = cache.get("key", "") != key
    if flag:

        isolated_trace.send((estimator, method))
        text = isolated_trace.recv()
        # if tracing does not function in isolated_trace, run it in parent process and error
        if text == "":
            sklearnex_trace(estimator, method)
            # guarantee failure if intermittent
            assert text, f"sklearnex_trace failure for {estimator}.{method}"

        for modulename, file in _TRACE_ALLOW_DICT.items():
            text = text.replace(file, modulename)
        regex_func = (
            r"(?<=funcname: )\S*(?=\n)"  # needed due to differences in module structure
        )
        regex_mod = r"(?<=--- modulename: )\S*(?=\.py)"  # needed due to differences in module structure

        regex_callingline = r"(?<=\n)\S.*(?=\n --- modulename: )"

        cache.set("key", key)
        cache.set(
            "text",
            {
                "funcs": re.findall(regex_func, text),
                "trace": text,
                "modules": [i.replace(os.sep, ".") for i in re.findall(regex_mod, text)],
                "callingline": [""] + re.findall(regex_callingline, text),
            },
        )

    return cache.get("text", None)


def call_validate_data(text, estimator, method):
    """test that both sklearnex wrapper for validate_data and
    original sklearn function/method validate_data are
    called once before offloading to oneDAL in sklearnex"""
    try:
        # get last to_table call showing end of oneDAL input portion of code
        idx = len(text["funcs"]) - 1 - text["funcs"][::-1].index("to_table")
        valid_funcs = text["funcs"][:idx]
        valid_modules = text["modules"][:idx]
    except ValueError:
        pytest.skip("onedal backend not used in this function")

    validate_data_calls = []
    for func, module in zip(valid_funcs, valid_modules):
        if func.endswith("validate_data"):
            validate_data_calls.append({module, func})

    assert (
        len(validate_data_calls) == 2
    ), "validate_data should be called two times: once for sklearn and once for sklearnex"
    assert validate_data_calls[0] == {
        "sklearnex.utils.validation",
        "validate_data",
    }, "sklearnex's validate_data should be called first"
    assert (
        (validate_data_calls[1] == {"sklearn.utils.validation", "validate_data"})
        if sklearn_check_version("1.6")
        else (validate_data_calls[1] == {"sklearn.base", "_validate_data"})
    ), "sklearn's validate_data should be called second"
    assert (
        valid_funcs.count("_check_feature_names") == 1
    ), "estimator should check feature names in validate_data"


def n_jobs_check(text, estimator, method):
    """verify the n_jobs is being set if '_get_backend' or 'to_table' is called"""
    # remove the _get_backend function from sklearnex from considered _get_backend
    count = max(
        text["funcs"].count("to_table"),
        len(
            [
                i
                for i in range(len(text["funcs"]))
                if text["funcs"][i] == "_get_backend"
                and "sklearnex" not in text["modules"][i]
            ]
        ),
    )
    n_jobs_count = text["funcs"].count("n_jobs_wrapper")

    assert bool(count) == bool(
        n_jobs_count
    ), f"verify if {method} should be in control_n_jobs' decorated_methods for {estimator}"


def runtime_property_check(text, estimator, method):
    """use of Python's 'property' should not be used at runtime, only at class instantiation"""
    assert (
        len(re.findall(r"property\(", text["trace"])) == 0
    ), f"{estimator}.{method} should only use 'property' at instantiation"


def fit_check_before_support_check(text, estimator, method):
    if "fit" not in method:
        if "dispatch" not in text["funcs"]:
            pytest.skip(f"onedal dispatching not used in {estimator}.{method}")
        idx = len(text["funcs"]) - 1 - text["funcs"][::-1].index("dispatch")
        validfuncs = text["funcs"][:idx]
        assert (
            "check_is_fitted" in validfuncs
        ), f"sklearn's check_is_fitted must be called before checking oneDAL support"

    else:
        pytest.skip(f"fitting occurs in {estimator}.{method}")


DESIGN_RULES = [
    n_jobs_check,
    runtime_property_check,
    fit_check_before_support_check,
    call_validate_data,
]


@pytest.mark.parametrize("design_pattern", DESIGN_RULES)
@pytest.mark.parametrize(
    "estimator, method",
    gen_models_info({**PATCHED_MODELS, **SPECIAL_INSTANCES}, fit=True, daal4py=False),
)
def test_estimator(estimator, method, design_pattern, estimator_trace):
    # These tests only apply to sklearnex estimators
    try:
        design_pattern(estimator_trace, estimator, method)
    except AssertionError:
        key = "-".join([estimator, method, design_pattern.__name__])
        if key in _DESIGN_RULE_VIOLATIONS:
            pytest.xfail(_DESIGN_RULE_VIOLATIONS[key])
        else:
            raise

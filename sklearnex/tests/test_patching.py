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


import importlib
import inspect
import io
import logging
import os
import re
import sys
from inspect import signature

import numpy as np
import numpy.random as nprnd
import pytest
from _utils import (
    DTYPES,
    PATCHED_FUNCTIONS,
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    UNPATCHED_FUNCTIONS,
    UNPATCHED_MODELS,
    gen_dataset,
    gen_models_info,
)

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import is_patched_instance
from sklearnex.dispatcher import _is_preview_enabled
from sklearnex.metrics import pairwise_distances, roc_auc_score


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues(dataframe_filter_="numpy")
)
@pytest.mark.parametrize("metric", ["cosine", "correlation"])
def test_pairwise_distances_patching(caplog, dataframe, queue, dtype, metric):
    with caplog.at_level(logging.WARNING, logger="sklearnex"):
        rng = nprnd.default_rng()
        X = _convert_to_dataframe(
            rng.random(size=1000), sycl_queue=queue, target_df=dataframe, dtype=dtype
        )

        _ = pairwise_distances(X.reshape(1, -1), metric=metric)
    assert all(
        [
            "running accelerated version" in i.message
            or "fallback to original Scikit-learn" in i.message
            for i in caplog.records
        ]
    ), f"sklearnex patching issue in pairwise_distances with log: \n{caplog.text}"


@pytest.mark.parametrize(
    "dtype", [i for i in DTYPES if "32" in i.__name__ or "64" in i.__name__]
)
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues(dataframe_filter_="numpy")
)
def test_roc_auc_score_patching(caplog, dataframe, queue, dtype):
    if dtype in DTYPES[-2:] and sys.platform == "win32":
        pytest.skip("Windows issue with unsigned ints")
    with caplog.at_level(logging.WARNING, logger="sklearnex"):
        rng = nprnd.default_rng()
        X = _convert_to_dataframe(
            rng.integers(2, size=1000),
            sycl_queue=queue,
            target_df=dataframe,
            dtype=dtype,
        )
        y = _convert_to_dataframe(
            rng.integers(2, size=1000),
            sycl_queue=queue,
            target_df=dataframe,
            dtype=dtype,
        )

        _ = roc_auc_score(X, y)
    assert all(
        [
            "running accelerated version" in i.message
            or "fallback to original Scikit-learn" in i.message
            for i in caplog.records
        ]
    ), f"sklearnex patching issue in roc_auc_score with log: \n{caplog.text}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues(dataframe_filter_="numpy")
)
@pytest.mark.parametrize("estimator, method", gen_models_info(PATCHED_MODELS))
def test_standard_estimator_patching(caplog, dataframe, queue, dtype, estimator, method):
    with caplog.at_level(logging.WARNING, logger="sklearnex"):
        est = PATCHED_MODELS[estimator]()

        if estimator == "TSNE" and method == "fit_transform":
            pytest.skip("TSNE.fit_transform is too slow for common testing")
        elif (
            estimator == "Ridge"
            and method in ["predict", "score"]
            and sys.platform == "win32"
            and dtype in DTYPES[-2:]
        ):
            pytest.skip("Windows segmentation fault for Ridge.predict for unsigned ints")
        elif not hasattr(est, method):
            pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")
        X, y = gen_dataset(est, queue=queue, target_df=dataframe, dtype=dtype)
        est.fit(X, y)

        if method != "score":
            getattr(est, method)(X)
        else:
            est.score(X, y)
    assert all(
        [
            "running accelerated version" in i.message
            or "fallback to original Scikit-learn" in i.message
            for i in caplog.records
        ]
    ), f"sklearnex patching issue in {estimator}.{method} with log: \n{caplog.text}"


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues(dataframe_filter_="numpy")
)
@pytest.mark.parametrize("estimator, method", gen_models_info(SPECIAL_INSTANCES))
def test_special_estimator_patching(caplog, dataframe, queue, dtype, estimator, method):
    # prepare logging

    with caplog.at_level(logging.WARNING, logger="sklearnex"):
        est = SPECIAL_INSTANCES[estimator]

        X, y = gen_dataset(est, queue=queue, target_df=dataframe, dtype=dtype)
        est.fit(X, y)

        if not hasattr(est, method):
            pytest.skip(f"sklearn available_if prevents testing {estimator}.{method}")
        if method != "score":
            getattr(est, method)(X)
        else:
            est.score(X, y)
    assert all(
        [
            "running accelerated version" in i.message
            or "fallback to original Scikit-learn" in i.message
            for i in caplog.records
        ]
    ), f"sklearnex patching issue in {estimator}.{method} with log: \n{caplog.text}"


@pytest.mark.parametrize("estimator", UNPATCHED_MODELS.keys())
def test_standard_estimator_signatures(estimator):
    est = PATCHED_MODELS[estimator]()
    unpatched_est = UNPATCHED_MODELS[estimator]()

    # all public sklearn methods should have signature matches in sklearnex

    unpatched_est_methods = [
        i
        for i in dir(unpatched_est)
        if not i.startswith("_") and not i.endswith("_") and hasattr(unpatched_est, i)
    ]
    for method in unpatched_est_methods:
        est_method = getattr(est, method)
        unpatched_est_method = getattr(unpatched_est, method)
        if callable(unpatched_est_method):
            regex = rf"(?:sklearn|daal4py)\S*{estimator}"  # needed due to differences in module structure
            patched_sig = re.sub(regex, estimator, str(signature(est_method)))
            unpatched_sig = re.sub(regex, estimator, str(signature(unpatched_est_method)))
            assert (
                unpatched_sig == patched_sig
            ), f"Signature of {estimator}.{method} does not match sklearn"


@pytest.mark.parametrize("function", UNPATCHED_FUNCTIONS.keys())
def test_patched_function_signatures(function):
    func = PATCHED_FUNCTIONS[function]
    unpatched_func = UNPATCHED_FUNCTIONS[function]

    if callable(unpatched_func):
        assert str(signature(func)) == str(
            signature(unpatched_func)
        ), f"Signature of {func} does not match sklearn"


def test_patch_map_match():
    # This rule applies to functions and classes which are out of preview.
    # Items listed in a matching submodule's __all__ attribute should be
    # in get_patch_map. There should not be any missing or additional elements.

    def list_all_attr(string):
        try:
            modules = set(importlib.import_module(string).__all__)
        except ModuleNotFoundError:
            modules = set([None])
        return modules

    if _is_preview_enabled():
        pytest.skip("preview sklearnex has been activated")
    patched = {**PATCHED_MODELS, **PATCHED_FUNCTIONS}

    sklearnex__all__ = list_all_attr("sklearnex")
    sklearn__all__ = list_all_attr("sklearn")

    module_map = {i: i for i in sklearnex__all__.intersection(sklearn__all__)}

    # _assert_all_finite and _logistic_regression_path patch internal
    # sklearn functions which aren't exposed. These are not available in
    # __all__ and require more careful anaylsis.

    for i in patched.copy():
        if i.startswith("_"):
            del patched[i]
    for module in module_map:
        sklearn_module__all__ = list_all_attr("sklearn." + module_map[module])
        sklearnex_module__all__ = list_all_attr("sklearnex." + module)
        intersect = sklearnex_module__all__.intersection(sklearn_module__all__)

        assert (
            intersect == sklearnex_module__all__
        ), f"{sklearnex_module__all__ - intersect} should not be in sklearnex.{module}.__all__"

        for i in intersect:
            if i:
                del patched[i]
            else:
                del patched[module]
    assert patched == {}, f"{patched.keys()} were not properly patched"


@pytest.mark.parametrize("estimator", UNPATCHED_MODELS.keys())
def test_is_patched_instance(estimator):
    patched = PATCHED_MODELS[estimator]
    unpatched = UNPATCHED_MODELS[estimator]
    assert is_patched_instance(patched), f"{patched} is a patched instance"
    assert not is_patched_instance(unpatched), f"{unpatched} is an unpatched instance"


@pytest.mark.parametrize("estimator", UNPATCHED_MODELS.keys())
def test_docstring_patching_match(estimator):
    patched = PATCHED_MODELS[estimator]
    unpatched = UNPATCHED_MODELS[estimator]
    patched_docstrings = {
        i: getattr(patched, i).__doc__
        for i in dir(patched)
        if not i.startswith("_") and not i.endswith("_") and hasattr(patched, i)
    }
    unpatched_docstrings = {
        i: getattr(unpatched, i).__doc__
        for i in dir(unpatched)
        if not i.startswith("_") and not i.endswith("_") and hasattr(unpatched, i)
    }

    # check class docstring match if a docstring is available

    assert (patched.__doc__ is None) == (unpatched.__doc__ is None)

    # check class attribute docstrings

    for i in unpatched_docstrings:
        assert (patched_docstrings[i] is None) == (unpatched_docstrings[i] is None)


@pytest.mark.parametrize("member", ["_onedal_cpu_supported", "_onedal_gpu_supported"])
@pytest.mark.parametrize(
    "name",
    [i for i in PATCHED_MODELS.keys() if "sklearnex" in PATCHED_MODELS[i].__module__],
)
def test_onedal_supported_member(name, member):
    patched = PATCHED_MODELS[name]
    sig = str(inspect.signature(getattr(patched, member)))
    assert "(self, method_name, *data)" == sig

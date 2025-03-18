# ==============================================================================
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
# ==============================================================================

import inspect
import logging
import os

import pytest
from joblib import cpu_count
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from threadpoolctl import threadpool_info

from sklearnex.tests.utils import PATCHED_MODELS, SPECIAL_INSTANCES, call_method

_X, _Y = make_classification(n_samples=40, n_features=4, random_state=42)


def _get_estimator_instance(estimator):
    if estimator in PATCHED_MODELS:
        est = PATCHED_MODELS[estimator]()
    elif estimator in SPECIAL_INSTANCES:
        est = SPECIAL_INSTANCES[estimator]
    else:
        raise KeyError(f"{estimator} not in patch_map or SPECIAL_INSTANCES")
    return est


def _check_n_jobs_entry_in_logs(records, function_name, n_jobs):
    expected_n_jobs = max(n_jobs, n_jobs % (cpu_count() + 1)) if n_jobs else cpu_count()
    for rec in records:
        if f"{function_name}: setting {expected_n_jobs} threads" in rec:
            return True
    # False if n_jobs is set and not found in logs
    return n_jobs is None or expected_n_jobs == cpu_count()


@pytest.mark.parametrize("estimator", {**PATCHED_MODELS, **SPECIAL_INSTANCES}.keys())
def test_n_jobs_documentation(estimator):
    est = _get_estimator_instance(estimator)
    assert "n_jobs" in est.__doc__
    assert "n_jobs" in est.__class__.__doc__


@pytest.mark.parametrize("estimator", {**PATCHED_MODELS, **SPECIAL_INSTANCES}.keys())
def test_n_jobs_method_decoration(estimator):
    est = _get_estimator_instance(estimator)
    for func_name, func in vars(est).items():
        # hasattr check necessary due to sklearn's available_if wrapper
        if hasattr(est, func_name) and callable(func):
            assert hasattr(func, "__onedal_n_jobs_decorated__") == (
                func_name in est._n_jobs_supported_onedal_methods
            ), f"{est}.{func_name} n_jobs decoration does not match {est} n_jobs supported methods"


@pytest.mark.parametrize("estimator", {**PATCHED_MODELS, **SPECIAL_INSTANCES}.keys())
@pytest.mark.parametrize("n_jobs", [None, -1, 1, 2])
def test_n_jobs_support(estimator, n_jobs, caplog):

    est = _get_estimator_instance(estimator)
    caplog.set_level(logging.DEBUG, logger="sklearnex")

    # copy params and modify n_jobs, assumes estimator inherits from BaseEstimator
    # or properly supports get_params and set_params methods as defined by sklearn
    params = est.get_params()
    params["n_jobs"] = n_jobs
    est.set_params(**params)

    # check `n_jobs` log entry for supported methods
    # `fit` call is required before other methods
    est.fit(_X, _Y)
    messages = [msg.message for msg in caplog.records]
    assert _check_n_jobs_entry_in_logs(messages, "fit", n_jobs)

    for method_name in est._n_jobs_supported_onedal_methods:
        # do not call fit again, handle sklearn's available_if wrapper
        if method_name == "fit" or (
            "NearestNeighbors" in estimator and "radius" in method_name
        ):
            # radius_neighbors and radius_neighbors_graph violate sklearn fallback guard
            # but use sklearnex interally, additional development must be done to those
            # functions to bring them to design compliance.
            continue
        try:
            call_method(est, method_name, _X, _Y)
        except (NotFittedError, AttributeError) as e:
            # handle sklearns available_if wrapper
            continue

        messages = [msg.message for msg in caplog.records]
        assert _check_n_jobs_entry_in_logs(messages, method_name, n_jobs)


@pytest.mark.skipif(
    not hasattr(os, "sched_setaffinity") or len(os.sched_getaffinity(0)) < 2,
    reason="python CPU affinity control unavailable or too few threads",
)
@pytest.mark.parametrize("estimator", {**PATCHED_MODELS, **SPECIAL_INSTANCES}.keys())
def test_n_jobs_affinity(estimator, caplog):
    # verify that n_jobs 1) starts at default value of cpu_count
    # 2) respects os.sched_setaffinity on supported machines
    n_t = next(i for i in threadpool_info() if i["user_api"] == "oneDAL")["num_threads"]

    # get affinity mask of calling process
    mask = os.sched_getaffinity(0)
    # by default, oneDAL should match the number of threads made available to the sklearnex pytest suite
    assert len(mask) == n_t

    try:
        # use half of the available threads
        newmask = set(list(mask)[: len(mask) // 2])
        os.sched_setaffinity(0, newmask)
        test_n_jobs_support(estimator, None, caplog)

    finally:
        # reset affinity mask no matter what
        os.sched_setaffinity(0, mask)

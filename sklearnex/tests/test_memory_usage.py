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

import gc
import logging
import os
import tracemalloc
import warnings
from inspect import isclass

import numpy as np
import pytest
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator, clone
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

from onedal import _default_backend as backend
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from onedal.tests.utils._device_selection import get_queues, is_dpctl_device_available
from onedal.utils._array_api import _get_sycl_namespace
from sklearnex import config_context
from sklearnex.tests.utils import (
    PATCHED_FUNCTIONS,
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    DummyEstimator,
)

CPU_SKIP_LIST = (
    "TSNE",  # too slow for using in testing on common data size
    "config_context",  # does not malloc
    "get_config",  # does not malloc
    "set_config",  # does not malloc
    "SVC(probability=True)",  # memory leak fortran numpy (investigate _fit_proba)
    "NuSVC(probability=True)",  # memory leak fortran numpy (investigate _fit_proba)
    "IncrementalEmpiricalCovariance",  # dataframe_f issues
    "IncrementalLinearRegression",  # TODO fix memory leak issue in private CI for data_shape = (1000, 100), data_transform_function = dataframe_f
    "IncrementalPCA",  # TODO fix memory leak issue in private CI for data_shape = (1000, 100), data_transform_function = dataframe_f
    "IncrementalRidge",  # TODO fix memory leak issue in private CI for data_shape = (1000, 100), data_transform_function = dataframe_f
    "LogisticRegression(solver='newton-cg')",  # memory leak fortran (1000, 100)
)

GPU_SKIP_LIST = (
    "TSNE",  # too slow for using in testing on common data size
    "RandomForestRegressor",  # too slow for using in testing on common data size
    "KMeans",  # does not support GPU offloading
    "config_context",  # does not malloc
    "get_config",  # does not malloc
    "set_config",  # does not malloc
    "ElasticNet",  # does not support GPU offloading (fails silently)
    "Lasso",  # does not support GPU offloading (fails silently)
    "SVR",  # does not support GPU offloading (fails silently)
    "NuSVR",  # does not support GPU offloading (fails silently)
    "NuSVC",  # does not support GPU offloading (fails silently)
    "LogisticRegression",  # default parameters not supported, see solver=newton-cg
    "NuSVC(probability=True)",  # does not support GPU offloading (fails silently)
    "IncrementalLinearRegression",  # issue with potrf with the specific dataset
    "LinearRegression",  # issue with potrf with the specific dataset
)


def gen_functions(functions):
    func_dict = functions.copy()

    roc_auc_score = func_dict.pop("roc_auc_score")
    func_dict["roc_auc_score"] = lambda x, y: roc_auc_score(y, y)

    pairwise_distances = func_dict.pop("pairwise_distances")
    func_dict["pairwise_distances(metric='cosine')"] = lambda x, y: pairwise_distances(
        x, metric="cosine"
    )
    func_dict["pairwise_distances(metric='correlation')"] = (
        lambda x, y: pairwise_distances(x, metric="correlation")
    )

    _assert_all_finite = func_dict.pop("_assert_all_finite")
    func_dict["_assert_all_finite"] = lambda x, y: [
        _assert_all_finite(x),
        _assert_all_finite(y),
    ]
    return func_dict


FUNCTIONS = gen_functions(PATCHED_FUNCTIONS)

CPU_ESTIMATORS = {
    k: v
    for k, v in {**PATCHED_MODELS, **SPECIAL_INSTANCES, **FUNCTIONS}.items()
    if not k in CPU_SKIP_LIST
}

GPU_ESTIMATORS = {
    k: v
    for k, v in {**PATCHED_MODELS, **SPECIAL_INSTANCES}.items()
    if not k in GPU_SKIP_LIST
}

data_shapes = [
    pytest.param((1000, 100), id="(1000, 100)"),
    pytest.param((2000, 50), id="(2000, 50)"),
]

EXTRA_MEMORY_THRESHOLD = 0.15
EXTRA_MEMORY_THRESHOLD_PANDAS = 0.25
N_SPLITS = 10
ORDER_DICT = {"F": np.asfortranarray, "C": np.ascontiguousarray}


def gen_clsf_data(n_samples, n_features, dtype=None):
    data, label = make_classification(
        n_classes=2, n_samples=n_samples, n_features=n_features, random_state=777
    )
    if dtype:
        data, label = data.astype(dtype), label.astype(dtype)
    return (
        data,
        label,
        data.size * data.dtype.itemsize + label.size * label.dtype.itemsize,
    )


def get_traced_memory(queue=None):
    if backend.is_dpc and queue and queue.sycl_device.is_gpu:
        return backend.get_used_memory(queue)
    else:
        return tracemalloc.get_traced_memory()[0]


def take(x, index, axis=0, queue=None):
    sycl_usm, xp, _ = _get_sycl_namespace(x)
    if sycl_usm:
        # Using the same sycl queue for dpnp.ndarray or usm_ndarray.
        return xp.take(
            x, xp.asarray(index, usm_type="device", sycl_queue=x.sycl_queue), axis=axis
        )
    elif hasattr(x, "__array_namespace__"):
        # check explicitly instead of sklearn's `get_namespace` as array_api is off by default
        xp = x.__array_namespace__()
        return xp.take(x, xp.asarray(index, device=x.device), axis=axis)
    else:
        return x.take(index, axis=axis)


def split_train_inference(kf, x, y, estimator, queue=None):
    mem_tracks = []
    for train_index, test_index in kf.split(x):
        x_train = take(x, train_index, queue=queue)
        y_train = take(y, train_index, queue=queue)
        x_test = take(x, test_index, queue=queue)
        y_test = take(y, test_index, queue=queue)

        if isclass(estimator) and issubclass(estimator, BaseEstimator):
            alg = estimator()
            flag = True
        elif isinstance(estimator, BaseEstimator):
            alg = clone(estimator)
            flag = True
        else:
            flag = False

        if flag:
            alg.fit(x_train, y_train)
            if hasattr(alg, "predict"):
                alg.predict(x_test)
            elif hasattr(alg, "transform"):
                alg.transform(x_test)
            elif hasattr(alg, "kneighbors"):
                alg.kneighbors(x_test)
            del alg
        else:
            estimator(x_train, y_train)

        del x_train, x_test, y_train, y_test, flag
        mem_tracks.append(get_traced_memory(queue))
    return mem_tracks


def _kfold_function_template(
    estimator, dataframe, data_shape, queue=None, func=None, dtype=None
):
    tracemalloc.start()

    n_samples, n_features = data_shape
    X, y, data_memory_size = gen_clsf_data(n_samples, n_features, dtype=dtype)
    kf = KFold(n_splits=N_SPLITS)
    if func:
        X = func(X)

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    mem_before = get_traced_memory(queue)
    mem_tracks = split_train_inference(kf, X, y, estimator, queue=queue)
    mem_iter_diffs = np.array(mem_tracks[1:]) - np.array(mem_tracks[:-1])
    mem_incr_mean, mem_incr_std = mem_iter_diffs.mean(), mem_iter_diffs.std()
    mem_incr_mean, mem_incr_std = round(mem_incr_mean), round(mem_incr_std)
    with warnings.catch_warnings():
        # In the case that the memory usage is constant, this will raise
        # a ConstantInputWarning error in pearsonr from scipy, this can
        # be ignored.
        warnings.filterwarnings(
            "ignore",
            message="An input array is constant; the correlation coefficient is not defined",
        )
        mem_iter_corr, _ = pearsonr(mem_tracks, list(range(len(mem_tracks))))

    if mem_iter_corr > 0.95:
        logging.warning(
            "Memory usage is steadily increasing with iterations "
            "(Pearson correlation coefficient between "
            f"memory tracks and iterations is {mem_iter_corr})\n"
            "Memory usage increase per iteration: "
            f"{mem_incr_mean}±{mem_incr_std} bytes"
        )
    mem_before_gc = get_traced_memory(queue)
    mem_diff = mem_before_gc - mem_before
    if isinstance(estimator, BaseEstimator):
        name = str(estimator)
    else:
        name = estimator.__name__

    threshold = (
        EXTRA_MEMORY_THRESHOLD_PANDAS if dataframe == "pandas" else EXTRA_MEMORY_THRESHOLD
    )
    message = (
        "Size of extra allocated memory {} using garbage collector "
        f"is greater than {threshold * 100}% of input data"
        f"\n\tAlgorithm: {name}"
        f"\n\tInput data size: {data_memory_size} bytes"
        "\n\tExtra allocated memory size: {} bytes"
        " / {} %"
    )
    if mem_diff >= threshold * data_memory_size:
        logging.warning(
            message.format(
                "before", mem_diff, round((mem_diff) / data_memory_size * 100, 2)
            )
        )
    gc.collect()
    mem_after = get_traced_memory(queue)
    tracemalloc.stop()
    mem_diff = mem_after - mem_before

    # GPU offloading with SYCL contains a program/kernel cache which should
    # be controllable via a KernelProgramCache object in the SYCL context.
    # The programs and kernels are stored on the GPU, but cannot be cleared
    # as this class is not available for access in all oneDAL DPC++ runtimes.
    # Therefore, until this is implemented this test must be skipped for gpu
    # as it looks like a memory leak (at least there is no way to discern a
    # leak on the first run).
    if queue is None or queue.sycl_device.is_cpu:
        assert mem_diff < threshold * data_memory_size, message.format(
            "after", mem_diff, round((mem_diff) / data_memory_size * 100, 2)
        )


@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("numpy,pandas,dpctl", "cpu")
)
@pytest.mark.parametrize("estimator", CPU_ESTIMATORS.keys())
@pytest.mark.parametrize("data_shape", data_shapes)
def test_memory_leaks(estimator, dataframe, queue, order, data_shape):
    func = ORDER_DICT[order]
    if estimator == "_assert_all_finite" and queue is not None:
        pytest.skip(f"{estimator} is not designed for device offloading")

    _kfold_function_template(
        CPU_ESTIMATORS[estimator], dataframe, data_shape, queue, func
    )


@pytest.mark.skipif(
    os.getenv("ZES_ENABLE_SYSMAN") is None or not is_dpctl_device_available(["gpu"]),
    reason="SYCL device memory leak check requires the level zero sysman",
)
@pytest.mark.parametrize("queue", get_queues("gpu"))
@pytest.mark.parametrize("estimator", GPU_ESTIMATORS.keys())
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("data_shape", data_shapes)
def test_gpu_memory_leaks(estimator, queue, order, data_shape):
    func = ORDER_DICT[order]
    if "ExtraTrees" in estimator and data_shape == (2000, 50):
        pytest.skip("Avoid a segmentation fault in Extra Trees algorithms")

    with config_context(target_offload=queue):
        _kfold_function_template(GPU_ESTIMATORS[estimator], None, data_shape, queue, func)


@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp,array_api", "cpu,gpu")
)
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("data_shape", data_shapes)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_table_conversions_memory_leaks(dataframe, queue, order, data_shape, dtype):
    func = ORDER_DICT[order]

    if (
        queue
        and queue.sycl_device.is_gpu
        and (
            os.getenv("ZES_ENABLE_SYSMAN") is None
            or not is_dpctl_device_available(["gpu"])
        )
    ):
        pytest.skip("SYCL device memory leak check requires the level zero sysman")

    _kfold_function_template(
        DummyEstimator,
        dataframe,
        data_shape,
        queue,
        func,
        dtype,
    )

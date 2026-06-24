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
import logging
import os
import pickle
import re
import sys
from functools import partial
from inspect import signature

import numpy as np
import numpy.random as nprnd
import pytest
from scipy import sparse as sp
from sklearn.base import BaseEstimator, ClusterMixin, RegressorMixin
from sklearn.svm._base import BaseLibSVM

from daal4py.sklearn._utils import (
    _package_check_version,
    is_sparse,
    sklearn_check_version,
)

if sklearn_check_version("1.6"):
    from sklearn.base import is_clusterer, is_regressor
else:

    def is_regressor(est):
        return isinstance(est, RegressorMixin)

    def is_clusterer(est):
        return isinstance(est, ClusterMixin)


from sklearn import get_config as sklearn_get_config

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context, is_patched_instance
from sklearnex._utils import get_tags
from sklearnex.dispatcher import _is_preview_enabled
from sklearnex.metrics import pairwise_distances, roc_auc_score
from sklearnex.tests.utils import (
    DTYPES,
    PATCHED_FUNCTIONS,
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    UNPATCHED_FUNCTIONS,
    UNPATCHED_MODELS,
    call_method,
    check_is_dynamic_method,
    gen_dataset,
    gen_models_info,
)
from sklearnex.utils._array_api import get_namespace


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues("numpy,pandas", "cpu")
)
@pytest.mark.parametrize("metric", ["cosine", "correlation"])
def test_pairwise_distances_patching(caplog, dataframe, queue, dtype, metric):
    with caplog.at_level(logging.WARNING, logger="sklearnex"):
        rng = nprnd.default_rng(seed=123)
        if dataframe == "pandas":
            X = _convert_to_dataframe(
                rng.random(size=1000).astype(dtype).reshape(1, -1),
                target_df=dataframe,
            )
        else:
            X = _convert_to_dataframe(
                rng.random(size=1000), sycl_queue=queue, target_df=dataframe, dtype=dtype
            )[None, :]

        _ = pairwise_distances(X, metric=metric)
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
    "dataframe, queue", get_dataframes_and_queues("numpy,pandas", "cpu")
)
def test_roc_auc_score_patching(caplog, dataframe, queue, dtype):
    if dtype in [np.uint32, np.uint64] and sys.platform == "win32":
        pytest.skip("Windows issue with unsigned ints")

    with caplog.at_level(logging.WARNING, logger="sklearnex"):
        rng = nprnd.default_rng(seed=123)
        X = rng.integers(2, size=1000)
        y = rng.integers(2, size=1000)

        X = _convert_to_dataframe(
            X,
            sycl_queue=queue,
            target_df=dataframe,
            dtype=dtype,
        )
        y = _convert_to_dataframe(
            y,
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


def _check_estimator_patching(caplog, dataframe, queue, dtype, est, method):
    # This should be modified as more array_api frameworks are tested and for
    # upcoming changes in dpnp

    result = None
    with caplog.at_level(logging.WARNING, logger="sklearnex"):
        from sklearn.datasets import load_breast_cancer

        if est.__class__.__name__ in ["LogisticRegression", "LogisticRegressionCV"]:
            # For LogisticRegression only binary classification is supported
            binary_dataset_dict = {
                "classification": [partial(load_breast_cancer, return_X_y=True)]
            }
            X, y = gen_dataset(
                est,
                queue=queue,
                target_df=dataframe,
                dtype=dtype,
                datasets=binary_dataset_dict,
            )[0]
        else:
            X, y = gen_dataset(est, queue=queue, target_df=dataframe, dtype=dtype)[0]

        est.fit(X, y)
        if method:
            if not hasattr(est, method) and check_is_dynamic_method(est, method):
                pytest.skip(f"sklearn available_if prevents testing {est}.{method}")
            result = call_method(est, method, X, y)

    assert all(
        [
            "running accelerated version" in i.message
            or "fallback to original Scikit-learn" in i.message
            for i in caplog.records
        ]
    ), f"sklearnex patching issue in {est}.{method} with log: \n{caplog.text}"

    return result, X, y


def _check_output_type(result, y, method, estimator_name, caplog, X, est=None):
    """Check output type, device, and dtype conformance.

    Checks for each result element:
      1. Sparse: check class -> skip
      2. Type: assert isinstance(res, input_type_y)
      3. Device: assert res.device == X.device
      4. Dtype: assert res.dtype == y.dtype (predict) or X.dtype (other)

    Skipped when: fell_back, _SKIP(output_type), _SKIP(output_dtype),
    regressor/clusterer predict, SVM decision_function, sparse, scalar.
    est=None for standalone functions (e.g. pairwise_distances).
    """
    # Automated numpy-OK checks based on estimator type / method
    if est is not None and is_clusterer(est) and method == "fit_predict":
        # ClusterMixin.fit_predict returns self.labels_ (numpy)
        return
    if method == "apply":
        # Tree apply returns integer leaf indices (numpy)
        return
    # Remaining known exceptions where numpy output is acceptable
    if _should_skip(estimator_name, method, "output_type"):
        return

    # Methods that return self (e.g. partial_fit) are not array outputs
    if isinstance(result, BaseEstimator):
        return

    input_type_y = type(y)

    # Check if sklearn fallback occurred (any record in caplog)
    fell_back = any(
        "fallback to original Scikit-learn" in r.message for r in caplog.records
    )

    # Collect all array results (handle tuples like kneighbors)
    if isinstance(result, (tuple, list)):
        results_to_check = result
    else:
        results_to_check = (result,)

    for res in results_to_check:
        if res is None:
            continue
        # Skip scalar results
        if np.isscalar(res):
            continue
        # Sparse outputs — verify sparse class matches sklearn config
        if is_sparse(res):
            if sklearn_check_version("1.9"):
                sparse_iface = sklearn_get_config().get("sparse_interface", "spmatrix")
                if sparse_iface == "sparray":
                    assert isinstance(res, sp.sparray)
                else:
                    assert isinstance(res, sp.spmatrix)
            continue

        if fell_back:
            # Fallback to sklearn: numpy output is acceptable
            assert isinstance(res, (np.ndarray, input_type_y))
        else:
            # Accelerated version: output must match input type
            assert isinstance(res, input_type_y)

            # Check device alignment: if input was on GPU, output
            # should also be on the same device. Uses standard array API
            # `.device` attribute for compatibility with torch, dpnp, etc.
            if hasattr(X, "device"):
                assert hasattr(res, "device")
                assert X.device == res.device

            # Check dtype preservation (skip float16 — oneDAL doesn't
            # support it natively and upcasts to float64)
            x_is_fp16 = (
                X is not None and hasattr(X, "dtype") and "float16" in str(X.dtype)
            )
            # Clusterers always return int cluster labels
            # decision_path returns structural int output — skip dtype
            if method == "decision_path":
                continue
            _skip_dtype = (
                method == "predict"
                and est is not None
                and (
                    is_clusterer(est)
                    or (is_regressor(est) and "int" in str(getattr(y, "dtype", "")))
                )
            )
            if (
                hasattr(res, "dtype")
                and not is_sparse(res)
                and not x_is_fp16
                and not _skip_dtype
                and not _should_skip(estimator_name, method, "output_dtype")
            ):
                if method == "predict" and y is not None and hasattr(y, "dtype"):
                    # predict output dtype should match y dtype
                    assert res.dtype == y.dtype
                elif X is not None and hasattr(X, "dtype") and "float" in str(X.dtype):
                    # Only check float results against float X dtype;
                    # int results (e.g. indices from kneighbors, labels
                    # from fit_predict) are expected to differ.
                    if "int" not in str(res.dtype):
                        assert res.dtype == X.dtype


# Unified skip rules: (estimator, name) -> set of checks to skip.
# Checks: output_type, output_dtype, attr_type, attr_device, attr_dtype.
_SKIP = {
    # Output
    ("DummyRegressor", "predict"): {"output_type"},
    ("NearestNeighbors", "radius_neighbors"): {"output_type"},
    ("ElasticNet", "path"): {"output_dtype"},
    ("Lasso", "path"): {"output_dtype"},
    ("IncrementalEmpiricalCovariance", "mahalanobis"): {"output_dtype"},
    # Attr — always
    ("DummyRegressor", "constant_"): {"attr_type", "attr_device", "attr_dtype"},
    # Attr — specific
    ("SVC", "n_iter_"): {"attr_device"},
    ("NuSVC", "n_iter_"): {"attr_device"},
    ("SVC", "probA_"): {"attr_device", "attr_dtype"},
    ("SVC", "probB_"): {"attr_device", "attr_dtype"},
    ("NuSVC", "probA_"): {"attr_device", "attr_dtype"},
    ("NuSVC", "probB_"): {"attr_device", "attr_dtype"},
    ("SVC", "class_weight_"): {"attr_dtype"},
    ("NuSVC", "class_weight_"): {"attr_dtype"},
}


def _should_skip(estimator_name, name, check):
    return check in _SKIP.get((estimator_name, name), set())


# Attrs that must be arrays — assert not scalar.
_ATTR_CHECK_MUST_BE_ARRAY = {
    "coef_",
    "intercept_",
    "dual_coef_",
    "support_vectors_",
    "cluster_centers_",
    "components_",
    "singular_values_",
    "explained_variance_",
    "explained_variance_ratio_",
    "mean_",
    "labels_",
    "class_weight_",
}


def _is_public_fitted_attr(attr_name, attr_val):
    """Check if attribute is a public fitted array attribute."""
    if not (attr_name[0].isalpha() and attr_name.endswith("_")):
        return False
    if not hasattr(attr_val, "dtype"):
        return False
    if np.isscalar(attr_val):
        return False
    return True


def _check_sparse_class(attr_val):
    """Verify sparse attr matches sklearn sparse_interface config."""
    if sklearn_check_version("1.9"):
        sparse_iface = sklearn_get_config().get("sparse_interface", "spmatrix")
        if sparse_iface == "sparray":
            assert isinstance(attr_val, sp.sparray)
        else:
            assert isinstance(attr_val, sp.spmatrix)


def _should_skip_all_for_attr(
    key, attr_name, est, is_non_numpy_input, fell_back, queue=None
):
    """Check if all type/device/dtype checks should be skipped for this attr."""
    # classes_ is numpy without dispatch, correct type with dispatch
    if attr_name == "classes_" and not sklearn_get_config().get(
        "array_api_dispatch", False
    ):
        return True
    if is_clusterer(est):
        return True
    if _should_skip(key[0], key[1], "attr_type"):
        return True
    if fell_back:
        return True
    return False


def _should_skip_dtype_for_attr(key, attr_name, est, x_is_fp16):
    """Check if dtype check should be skipped for this attr."""
    if isinstance(est, BaseLibSVM):
        return True
    if x_is_fp16:
        return True
    if _should_skip(key[0], key[1], "attr_dtype"):
        return True
    return False


def _check_fitted_attributes(est, X, estimator_name, caplog, queue=None):
    """Check fitted attributes preserve input type, device, and dtype.

    Call sequence for each attr:
      1. Filter: skip non-public, non-array, scalar
      2. Sparse: check class -> skip
      3. Skip: _SKIP dict (attr_type, attr_device, attr_dtype),
         classes_ (no dispatch), clusterers, fell_back -> skip
      4. Type: assert isinstance(attr, input_type)
      5. Device: assert attr.device == X.device (skip via _SKIP)
      6. Dtype: assert attr.dtype == X.dtype (skip via _SKIP,
         BaseLibSVM, fp16)
    """
    input_type = type(X)
    xp, _ = get_namespace(X)
    is_non_numpy_input = not isinstance(X, np.ndarray) and xp is not None

    fell_back = any(
        "fallback to original Scikit-learn" in r.message for r in caplog.records
    )

    x_is_fp16 = hasattr(X, "dtype") and "float16" in str(X.dtype)

    for attr_name, attr_val in vars(est).items():
        # Filter
        if not _is_public_fitted_attr(attr_name, attr_val):
            continue

        # Assert attrs that must be arrays are not scalar
        if attr_name in _ATTR_CHECK_MUST_BE_ARRAY:
            assert hasattr(attr_val, "ndim") and attr_val.ndim > 0

        # Sparse — check class then skip
        if is_sparse(attr_val):
            _check_sparse_class(attr_val)
            continue

        key = (estimator_name, attr_name)

        # Known exceptions — skip all
        if _should_skip_all_for_attr(
            key, attr_name, est, is_non_numpy_input, fell_back, queue
        ):
            if fell_back:
                assert isinstance(attr_val, (np.ndarray, input_type))
            continue

        # Type check
        assert isinstance(attr_val, input_type)

        # Device check
        if (
            not _should_skip(estimator_name, attr_name, "attr_device")
            and hasattr(X, "device")
            and hasattr(attr_val, "device")
        ):
            assert X.device == attr_val.device

        # Dtype check
        if not _should_skip_dtype_for_attr(key, attr_name, est, x_is_fp16):
            if (
                hasattr(attr_val, "dtype")
                and "float" in str(attr_val.dtype)
                and hasattr(X, "dtype")
                and "float" in str(X.dtype)
            ):
                assert attr_val.dtype == X.dtype


def _check_set_output_transform(est, method, X, estimator_name):
    """Test set_output(transform=...) for transform methods.

    Verifies that sklearn's set_output API is respected by sklearnex
    estimators. When set_output(transform="pandas") is configured,
    transform() should return a pandas DataFrame, etc.

    Only applies to the 'transform' method (not fit_transform to avoid
    refitting the estimator).
    """
    if method != "transform":
        return
    if not hasattr(est, "set_output"):
        return

    for transform_output in ["default", "pandas", "polars"]:
        est.set_output(transform=transform_output)
        try:
            result = est.transform(X)
        except Exception:
            # Some input types (e.g., dpnp, array_api_strict) may not
            # support conversion to the requested output format,
            # or the output library (e.g., polars) may not be installed
            continue

        if transform_output == "pandas":
            import pandas as pd

            expected_type = pd.DataFrame
        elif transform_output == "polars":
            import polars as pl

            expected_type = pl.DataFrame
        else:
            expected_type = None

        if expected_type is not None:
            assert isinstance(result, expected_type)

    # Reset to default
    est.set_output(transform="default")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
@pytest.mark.parametrize("estimator, method", gen_models_info(PATCHED_MODELS))
def test_standard_estimator_patching(caplog, dataframe, queue, dtype, estimator, method):
    kwargs = {}
    if (
        estimator == "LogisticRegression"
        and queue
        and getattr(queue.sycl_device, "is_gpu", False)
    ):
        # sklearnex.LogisticRegression only has support for newton-cg solver on GPU
        kwargs = {"solver": "newton-cg"}
    est = PATCHED_MODELS[estimator](**kwargs)

    if queue:
        if dtype == np.float16 and not queue.sycl_device.has_aspect_fp16:
            pytest.skip("Hardware does not support fp16 SYCL testing")
        elif dtype == np.float64 and not queue.sycl_device.has_aspect_fp64:
            pytest.skip("Hardware does not support fp64 SYCL testing")
        elif (
            (np.issubdtype(dtype, np.integer) or dtype == np.float16)
            and not queue.sycl_device.has_aspect_fp64
        ):
            # sklearn's check_array upcasts integer and float16 inputs to
            # float64 before passing them to the estimator, which fails on
            # fp64-less GPUs.
            pytest.skip("Hardware does not support fp64 SYCL testing")
        elif queue.sycl_device.is_gpu and estimator in [
            "ElasticNet",
            "Lasso",
        ]:
            pytest.skip(f"{estimator} does not support GPU queues")

    if "NearestNeighbors" in estimator and "radius" in method:
        pytest.skip(f"RadiusNeighbors estimator not implemented in sklearnex")

    if estimator == "TSNE" and method == "fit_transform":
        pytest.skip("TSNE.fit_transform is too slow for common testing")
    elif estimator == "IncrementalLinearRegression" and np.issubdtype(dtype, np.integer):
        pytest.skip(
            "IncrementalLinearRegression fails on oneDAL side with int types because dataset is filled by zeroes"
        )
    elif method and not hasattr(est, method) and not check_is_dynamic_method(est, method):
        pytest.skip(f"sklearn available_if prevents testing {est}.{method}")

    if (
        (dataframe == "array_api" or queue)
        and estimator == "LogisticRegressionCV"
        and (not sklearn_check_version("1.6") or not get_tags(est).array_api_support)
    ):
        pytest.skip("Array API and/or GPU inputs not supported in estimator")
    if (
        (
            dataframe == "array_api"
            or (
                dataframe in ["dpnp"]
                and (not queue or not getattr(queue.sycl_device, "is_gpu", False))
            )
        )
        and estimator == "LogisticRegression"
        and not sklearn_check_version("1.9")
    ):
        # In case array api inputs on CPU are provided on CPU for LogisticRegression sklearnex will fallback to sklearn
        # Array API support for LogisticRegression in sklearn is only available starting from 1.9
        pytest.skip(
            "Array API inputs on CPU are not supported for LogisticRegression in sklearn <1.9"
        )

    if dataframe == "array_api":
        # as array_api dispatching is experimental, sklearn support isn't guaranteed.
        # the infrastructure from sklearn that sklearnex depends on is also susceptible
        # to failure. In this case compare to sklearn for the same failure. By design
        # the patching of sklearn should act similarly. Technically this is conformance.
        if (
            (estimator == "PCA" and "transform" in method)
            or (estimator == "IncrementalEmpiricalCovariance" and method == "mahalanobis")
        ) and not _package_check_version("2.0", np.__version__):
            # issue not to be observed with normal numpy usage
            pytest.skip(
                f"numpy backend does not properly handle the __dlpack__ attribute."
            )
        elif (
            not sklearn_check_version("1.3")
            and estimator == "IncrementalEmpiricalCovariance"
            and method == "score"
        ):
            pytest.skip(
                f"array checking in sklearn <1.3 does not fully support array_api inputs, causes sklearnex-only estimator failure"
            )
        tags = get_tags(est)
        array_api_check = (
            hasattr(tags, "array_api_support") and tags.array_api_support
        ) or (hasattr(tags, "onedal_array_api") and tags.onedal_array_api)
        if not array_api_check:
            pytest.skip(
                "Array API support not implemented in either scikit-learn or scikit-learn-intelex"
            )

        with config_context(array_api_dispatch=True):
            try:
                result, X, y = _check_estimator_patching(
                    caplog, dataframe, queue, dtype, est, method
                )
            except Exception as e:
                # if we are borrowing from sklearn and it fails, then this is something
                # failing on sklearn-side. It is only allowed to fail if the underlying sklearn
                # function doesn't support array_api with the set parameters and array_api
                # support isn't promised by oneDAL
                if estimator not in UNPATCHED_MODELS or getattr(
                    PATCHED_MODELS[estimator], method
                ) != getattr(UNPATCHED_MODELS[estimator], method, None):
                    raise e
            else:
                # Check return type conformance when no exception
                # occurred. Output arrays should match the input array type.
                _check_output_type(result, y, method, estimator, caplog, X=X, est=est)
                _check_fitted_attributes(est, X, estimator, caplog, queue=queue)
                _check_set_output_transform(est, method, X, estimator)

    else:
        if dataframe == "dpnp":
            # Note: this tries to check for GPU support by checking for array API
            # support. If some class can run on GPU but doesn't support array API,
            # an exception should be made here.
            tags = get_tags(est)
            if not (hasattr(tags, "onedal_array_api") and tags.onedal_array_api):
                pytest.skip("No GPU support for estimator")
        result, X, y = _check_estimator_patching(
            caplog, dataframe, queue, dtype, est, method
        )
        # KNN/LOF store _fit_X as dpnp even without dispatch, so pickle
        # fails (dpnp SYCL queues are not serializable)
        if dataframe == "dpnp" and estimator not in [
            "KNeighborsClassifier",
            "KNeighborsRegressor",
            "NearestNeighbors",
            "LocalOutlierFactor",
        ]:

            pickle.loads(pickle.dumps(est))
        # Without array_api_dispatch, dpnp inputs go through
        # support_input_format (converts to numpy and back) for fit and
        # support_sycl_format (converts to numpy but NOT back) for some
        # methods like score_samples/mahalanobis. This creates a type
        # mismatch: fitted attrs may be numpy while method outputs are
        # dpnp or vice versa. With array_api_dispatch enabled, all paths
        # use array API consistently, so we re-fit and re-call with
        # dispatch on to verify output types correctly.
        if dataframe not in ("numpy", "pandas"):
            # Skip second pass if estimator doesn't support GPU for this data
            if queue is not None and getattr(queue.sycl_device, "is_gpu", False):
                X_np, y_np = _as_numpy(X), _as_numpy(y)
                if not est._onedal_gpu_supported("fit", X_np, y_np, None).get_status():
                    _check_set_output_transform(est, method, X, estimator)
                    return
            with config_context(array_api_dispatch=True):
                result2, X2, y2 = _check_estimator_patching(
                    caplog, dataframe, queue, dtype, est, method
                )
                _check_output_type(result2, y2, method, estimator, caplog, X=X2, est=est)
                _check_fitted_attributes(est, X2, estimator, caplog, queue=queue)
        _check_set_output_transform(est, method, X, estimator)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
@pytest.mark.parametrize("estimator, method", gen_models_info(SPECIAL_INSTANCES))
def test_special_estimator_patching(caplog, dataframe, queue, dtype, estimator, method):
    est = SPECIAL_INSTANCES[estimator]

    if queue:
        # Its not possible to get the dpnp arrays to be in the proper dtype
        if dtype == np.float16 and not queue.sycl_device.has_aspect_fp16:
            pytest.skip("Hardware does not support fp16 SYCL testing")
        elif dtype == np.float64 and not queue.sycl_device.has_aspect_fp64:
            pytest.skip("Hardware does not support fp64 SYCL testing")
        elif (
            (np.issubdtype(dtype, np.integer) or dtype == np.float16)
            and not queue.sycl_device.has_aspect_fp64
        ):
            # sklearn's check_array upcasts integer and float16 inputs to
            # float64 before passing them to the estimator, which fails on
            # fp64-less GPUs.
            pytest.skip("Hardware does not support fp64 SYCL testing")

    if "NearestNeighbors" in estimator and "radius" in method:
        pytest.skip(f"RadiusNeighbors estimator not implemented in sklearnex")

    _check_estimator_patching(caplog, dataframe, queue, dtype, est, method)


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
                patched_sig == unpatched_sig
            ), f"Signature of {estimator}.{method} does not match sklearn"


@pytest.mark.parametrize("estimator", UNPATCHED_MODELS.keys())
def test_standard_estimator_init_signatures(estimator):
    # Several estimators have additional parameters that are user-accessible
    # which are sklearnex-specific. They will fail and are removed from tests.
    # remove n_jobs due to estimator patching for sklearnex (known deviation)
    patched_sig = str(signature(PATCHED_MODELS[estimator].__init__))
    unpatched_sig = str(signature(UNPATCHED_MODELS[estimator].__init__))

    # Sklearnex allows for positional kwargs and n_jobs, when sklearn doesn't
    for kwarg in ["n_jobs=None", "*"]:
        patched_sig = patched_sig.replace(", " + kwarg, "")
        unpatched_sig = unpatched_sig.replace(", " + kwarg, "")

    # Special sklearnex-specific kwargs are removed from signatures here
    if estimator in [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "ExtraTreesRegressor",
        "ExtraTreesClassifier",
    ]:
        for kwarg in ["min_bin_size=1", "max_bins=256"]:
            patched_sig = patched_sig.replace(", " + kwarg, "")

    assert (
        patched_sig == unpatched_sig
    ), f"Signature of {estimator}.__init__ does not match sklearn"


@pytest.mark.parametrize(
    "function",
    [
        i
        for i in UNPATCHED_FUNCTIONS.keys()
        if i not in ["train_test_split", "set_config", "config_context"]
    ],
)
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
            mod = importlib.import_module(string)
        except ModuleNotFoundError:
            return set([None])

        # Some sklearn estimators exist in python
        # files rather than folders under sklearn
        modules = set(
            getattr(
                mod, "__all__", [name for name in dir(mod) if not name.startswith("_")]
            )
        )
        return modules

    if _is_preview_enabled():
        pytest.skip("preview sklearnex has been activated")
    patched = {**PATCHED_MODELS, **PATCHED_FUNCTIONS}

    sklearnex__all__ = list_all_attr("sklearnex")
    sklearn__all__ = list_all_attr("sklearn")

    module_map = {i: i for i in sklearnex__all__.intersection(sklearn__all__)}

    # remove all scikit-learn-intelex-only estimators
    for i in patched.copy():
        if i not in UNPATCHED_MODELS and i not in UNPATCHED_FUNCTIONS:
            del patched[i]

    for module in module_map:
        sklearn_module__all__ = list_all_attr("sklearn." + module_map[module])
        sklearnex_module__all__ = list_all_attr("sklearnex." + module)
        intersect = sklearnex_module__all__.intersection(sklearn_module__all__)

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


@pytest.mark.parametrize("estimator", PATCHED_MODELS.keys())
def test_if_estimator_inherits_sklearn(estimator):
    est = PATCHED_MODELS[estimator]
    if estimator in UNPATCHED_MODELS:
        assert issubclass(
            est, UNPATCHED_MODELS[estimator]
        ), f"{estimator} does not inherit from the patched sklearn estimator"
    else:
        assert issubclass(est, BaseEstimator)


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
    sig = str(signature(getattr(patched, member)))
    assert "(self, method_name, *data)" == sig

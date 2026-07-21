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

from contextlib import nullcontext

import array_api_strict
import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.datasets import load_iris

from daal4py.sklearn._utils import (
    _package_check_version,
    daal_check_version,
    sklearn_check_version,
)
from onedal import _dpc_backend
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    dpnp_available,
    get_dataframes_and_queues,
    torch_available,
    torch_xpu_available,
)
from onedal.tests.utils._device_selection import is_sycl_device_available
from sklearnex import config_context
from sklearnex.decomposition import PCA
from sklearnex.tests.utils import assert_transform_output_matches_default

if dpnp_available:
    import dpnp
if torch_available:
    import torch


@pytest.fixture
def hyperparameters(request):
    hparams = PCA.get_hyperparameters("fit")

    def restore_hyperparameters():
        if daal_check_version((2025, "P", 700)):
            PCA.reset_hyperparameters("fit")

    request.addfinalizer(restore_hyperparameters)
    return hparams


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("macro_block", [None, 2])
@pytest.mark.parametrize("grain_size", [None, 2])
def test_sklearnex_import(hyperparameters, dataframe, queue, macro_block, grain_size):

    X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    X_transformed_expected = [
        [-1.38340578, -0.2935787],
        [-2.22189802, 0.25133484],
        [-3.6053038, -0.04224385],
        [1.38340578, 0.2935787],
        [2.22189802, -0.25133484],
        [3.6053038, 0.04224385],
    ]

    if daal_check_version((2025, "P", 700)):
        if macro_block is not None:
            if queue and queue.sycl_device.is_gpu:
                pytest.skip("Test for CPU-only functionality")
            hyperparameters.cpu_macro_block = macro_block
        if grain_size is not None:
            if queue and queue.sycl_device.is_gpu:
                pytest.skip("Test for CPU-only functionality")
            hyperparameters.cpu_grain_size = grain_size

    pca = PCA(n_components=2, svd_solver="covariance_eigh")

    pca.fit(X)
    X_transformed = pca.transform(X)
    X_fit_transformed = PCA(n_components=2, svd_solver="covariance_eigh").fit_transform(X)

    if daal_check_version((2024, "P", 100)):
        assert "sklearnex" in pca.__module__
        assert hasattr(pca, "_onedal_estimator")
    else:
        assert "daal4py" in pca.__module__

    tol = 1e-5 if _as_numpy(X_transformed).dtype == np.float32 else 1e-7
    assert_allclose([6.30061232, 0.54980396], _as_numpy(pca.singular_values_))
    assert_allclose(X_transformed_expected, _as_numpy(X_transformed), rtol=tol)
    assert_allclose(X_transformed_expected, _as_numpy(X_fit_transformed), rtol=tol)


@pytest.mark.skipif(
    not daal_check_version((2025, "P", 700)),
    reason="Functionality introduced in a later version",
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_non_batched_covariance(hyperparameters, dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Test for CPU-only functionality")

    from sklearnex.decomposition import PCA

    # This generates a random matrix with non-independent columns
    rng = np.random.default_rng(seed=123)
    S = rng.standard_normal(size=(6, 5))
    S = S.T @ S
    mu = rng.standard_normal(size=5)
    X = rng.multivariate_normal(mu, S, size=20)

    hyperparameters.cpu_max_cols_batched = np.iinfo(np.int32).max
    res_batched = PCA().fit(X).components_

    hyperparameters.cpu_max_cols_batched = 1
    res_non_batched = PCA().fit(X).components_

    assert_allclose(res_non_batched, res_batched)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy"))
def test_changed_estimated_attributes(with_array_api, dataframe, queue):
    # check that attributes necessary for the PCA onedal estimator match
    # changes occurring in the sklearnex estimator
    X, y = load_iris(return_X_y=True)

    X_0 = _convert_to_dataframe(X[y == 0], sycl_queue=queue, target_df=dataframe)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    est0 = PCA(n_components=4).fit(X_0)
    est = clone(est0)
    assert not hasattr(est, "_onedal_estimator")
    est.fit(X)
    assert not np.array_equal(_as_numpy(est.transform(X)), _as_numpy(est0.transform(X)))

    # copy over parameters necessary for transform
    est.mean_ = est0.mean_
    est.components_ = est0.components_
    est.explained_variance_ = est0.explained_variance_
    est.n_components_ = est0.n_components_  # is trivial but exercises the logic

    assert np.array_equal(_as_numpy(est.transform(X)), _as_numpy(est0.transform(X)))


@pytest.mark.allow_sklearn_fallback
def test_create_model_behavior():
    # verify that fit fallbacks does not break ``transform`` as the oneDAL
    # model is generated JIT

    X, _ = load_iris(return_X_y=True)
    # generate a onedal estimator
    est = PCA(n_components=3)
    X_trans = est.fit_transform(X)

    # force data to sparse for a fallback to sklearn
    X_sp = sp.csr_matrix(X)
    est.fit(X_sp)
    # In the case of a fallback, the model should be set to none by clobbered
    # fitted attributes
    assert est._onedal_estimator._onedal_model is None

    X_trans_sparse = est.transform(X)
    # use allclose as data was fit with sklearn and onedal on the same data
    # but using different backends
    assert_allclose(X_trans, X_trans_sparse)


@pytest.mark.skipif(not dpnp_available, reason="Functionality to test requires DPNP.")
@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Relies on functionality introduced in later scikit-learn versions.",
)
@pytest.mark.skipif(
    not is_sycl_device_available("gpu"), reason="Test for GPU-specific functionality."
)
def test_pca_error_on_incompatible_devices(with_array_api):
    import dpnp

    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(20, 3), dtype=np.float32)
    X_cpu = dpnp.array(X, device="cpu")
    X_gpu = dpnp.array(X, device="gpu")

    err_match = "device|queue"

    model = PCA(svd_solver="covariance_eigh").fit(X_gpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.transform(X_cpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.inverse_transform(X_cpu)

    model.fit(X_cpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.transform(X_gpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.inverse_transform(X_gpu)


def _pca_convert(arr, xp, device):
    """Convert a numpy array to the array-API backend ``xp`` on ``device``."""
    if xp is np:
        return arr
    return xp.asarray(arr, device=device)


# array_api_strict output conversion fails on numpy < 2.2.5: PCA rebuilds its model
# from the fitted ``components_``, which numpy < 2.2.5 returns as a read-only array
# that ``to_table`` cannot export through DLPack (BufferError). numpy >= 2.2.5 returns
# a writeable array, so it works there.
# TODO: remove this skip once sklearnex handles read-only arrays in the oneDAL data
# conversion so array_api_strict works on numpy < 2.2.5 as well.
_PCA_ARRAY_API_STRICT = pytest.param(
    array_api_strict,
    None,
    marks=pytest.mark.skipif(
        not _package_check_version("2.2.5", np.__version__),
        reason="TODO: sklearnex read-only DLPack conversion fails on numpy<2.2.5",
    ),
)

# (xp, device) array-API input combinations, CPU and GPU; device-specific entries
# are dropped at collection time when the hardware/library is unavailable.
# dpnp arrays are SYCL arrays even on "cpu", so they need a SYCL-enabled sklearnex
# build (``_dpc_backend``) to be converted -- a CPU-only build raises "installation
# does not have SYCL support". ``is_sycl_device_available`` is not enough: it uses a
# dpctl queue that succeeds regardless of whether sklearnex was built with DPC.
_PCA_ARRAY_API_INPUTS = (
    [(np, None), _PCA_ARRAY_API_STRICT]
    + ([(dpnp, "cpu")] if dpnp_available and _dpc_backend is not None else [])
    + (
        [(dpnp, "gpu")]
        if dpnp_available and _dpc_backend is not None and is_sycl_device_available("gpu")
        else []
    )
    + ([(torch, "cpu")] if torch_available else [])
    + ([(torch, "xpu")] if torch_xpu_available else [])
)


@pytest.mark.parametrize("xp,device", _PCA_ARRAY_API_INPUTS)
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_transform_output_matches_default(
    xp, device, transform_output, method, with_array_api
):
    X = _pca_convert(load_iris(return_X_y=True)[0], xp, device)
    pca = PCA(n_components=3).fit(X)
    assert_transform_output_matches_default(pca, X, transform_output, method)


@pytest.mark.skipif(not dpnp_available, reason="Functionality to test requires DPNP.")
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("dpnp"))
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_transform_output_dpnp_no_array_api(dataframe, queue, transform_output, method):
    X = _convert_to_dataframe(
        load_iris(return_X_y=True)[0], sycl_queue=queue, target_df=dataframe
    )
    pca = PCA(n_components=3).fit(X)
    assert_transform_output_matches_default(pca, X, transform_output, method)


@pytest.mark.skipif(
    not is_sycl_device_available("gpu"), reason="Test for GPU-specific functionality."
)
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_transform_output_target_offload(transform_output, method):
    X = load_iris(return_X_y=True)[0]
    with config_context(target_offload="gpu"):
        pca = PCA(n_components=3).fit(X)
        assert_transform_output_matches_default(pca, X, transform_output, method)


@pytest.mark.parametrize("target_offload", [False, True])
@pytest.mark.parametrize("dataframe", [pd.DataFrame, pl.DataFrame])
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_transform_output_pandas_polars_input(
    dataframe, target_offload, transform_output, method
):
    if target_offload and not is_sycl_device_available("gpu"):
        pytest.skip("Test for GPU-specific functionality.")
    X = dataframe(load_iris(return_X_y=True)[0])
    ctx = config_context(target_offload="gpu") if target_offload else nullcontext()
    with ctx:
        pca = PCA(n_components=3).fit(X)
        assert_transform_output_matches_default(pca, X, transform_output, method)

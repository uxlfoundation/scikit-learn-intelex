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

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.datasets import load_iris

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.decomposition import PCA


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


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy,dpnp"))
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


@pytest.mark.skipif(
    not sklearn_check_version("1.5"),
    reason='svd_solver="auto" does not support sparse inputs',
)
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

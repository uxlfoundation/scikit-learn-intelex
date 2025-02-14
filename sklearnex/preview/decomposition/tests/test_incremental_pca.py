# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.preview.decomposition import IncrementalPCA


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import(dataframe, queue):
    X = [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    incpca = IncrementalPCA(n_components=2)
    result = incpca.fit(X)
    assert "sklearnex" in incpca.__module__
    assert hasattr(incpca, "_onedal_estimator")
    assert_allclose(_as_numpy(result.singular_values_), [6.30061232, 0.54980396])


def check_pca_on_gold_data(incpca, dtype, whiten, transformed_data):
    expected_n_samples_seen_ = 6
    expected_n_features_in_ = 2
    expected_n_components_ = 2
    expected_components_ = np.array([[0.83849224, 0.54491354], [-0.54491354, 0.83849224]])
    expected_singular_values_ = np.array([6.30061232, 0.54980396])
    expected_mean_ = np.array([0, 0])
    expected_var_ = np.array([5.6, 2.4])
    expected_explained_variance_ = np.array([7.93954312, 0.06045688])
    expected_explained_variance_ratio_ = np.array([0.99244289, 0.00755711])
    expected_noise_variance_ = 0.0
    expected_transformed_data = (
        np.array(
            [
                [-0.49096647, -1.19399271],
                [-0.78854479, 1.02218579],
                [-1.27951125, -0.17180692],
                [0.49096647, 1.19399271],
                [0.78854479, -1.02218579],
                [1.27951125, 0.17180692],
            ]
        )
        if whiten
        else np.array(
            [
                [-1.38340578, -0.2935787],
                [-2.22189802, 0.25133484],
                [-3.6053038, -0.04224385],
                [1.38340578, 0.2935787],
                [2.22189802, -0.25133484],
                [3.6053038, 0.04224385],
            ]
        )
    )

    tol = 1e-7
    if dtype == np.float32:
        tol = 7e-6 if whiten else 1e-6

    assert incpca.n_samples_seen_ == expected_n_samples_seen_
    assert incpca.n_features_in_ == expected_n_features_in_
    assert incpca.n_components_ == expected_n_components_

    assert_allclose(incpca.singular_values_, expected_singular_values_, atol=tol)
    assert_allclose(incpca.mean_, expected_mean_, atol=tol)
    assert_allclose(incpca.var_, expected_var_, atol=tol)
    assert_allclose(incpca.explained_variance_, expected_explained_variance_, atol=tol)
    assert_allclose(
        incpca.explained_variance_ratio_, expected_explained_variance_ratio_, atol=tol
    )
    assert np.abs(incpca.noise_variance_ - expected_noise_variance_) < tol
    if daal_check_version((2024, "P", 500)):
        assert_allclose(incpca.components_, expected_components_, atol=tol)
        assert_allclose(_as_numpy(transformed_data), expected_transformed_data, atol=tol)
    else:
        for i in range(incpca.n_components_):
            abs_dot_product = np.abs(
                np.dot(incpca.components_[i], expected_components_[i])
            )
            assert np.abs(abs_dot_product - 1.0) < tol

            if np.dot(incpca.components_[i], expected_components_[i]) < 0:
                assert_allclose(
                    _as_numpy(-transformed_data[i]),
                    expected_transformed_data[i],
                    atol=tol,
                )
            else:
                assert_allclose(
                    _as_numpy(transformed_data[i]), expected_transformed_data[i], atol=tol
                )


def check_pca(incpca, dtype, whiten, data, transformed_data):
    tol = 3e-3 if dtype == np.float32 else 2e-6

    n_components = incpca.n_components_

    expected_n_samples_seen = data.shape[0]
    expected_n_features_in = data.shape[1]
    n_samples_seen = incpca.n_samples_seen_
    n_features_in = incpca.n_features_in_
    assert n_samples_seen == expected_n_samples_seen
    assert n_features_in == expected_n_features_in

    components = incpca.components_
    singular_values = incpca.singular_values_
    centered_data = data - np.mean(data, axis=0)
    cov_eigenvalues, cov_eigenvectors = np.linalg.eig(
        centered_data.T @ centered_data / (n_samples_seen - 1)
    )
    cov_eigenvalues = np.nan_to_num(cov_eigenvalues)
    cov_eigenvalues[cov_eigenvalues < 0] = 0
    eigenvalues_order = np.argsort(cov_eigenvalues)[::-1]
    sorted_eigenvalues = cov_eigenvalues[eigenvalues_order]
    sorted_eigenvectors = cov_eigenvectors[:, eigenvalues_order]
    expected_singular_values = np.sqrt(sorted_eigenvalues * (n_samples_seen - 1))[
        :n_components
    ]
    expected_components = sorted_eigenvectors.T[:n_components]

    assert_allclose(singular_values, expected_singular_values, atol=tol)
    for i in range(n_components):
        component_length = np.dot(components[i], components[i])
        assert np.abs(component_length - 1.0) < tol
        abs_dot_product = np.abs(np.dot(components[i], expected_components[i]))
        assert np.abs(abs_dot_product - 1.0) < tol

    expected_mean = np.mean(data, axis=0)
    assert_allclose(incpca.mean_, expected_mean, atol=tol)

    expected_var = np.var(_as_numpy(data), ddof=1, axis=0)
    assert_allclose(incpca.var_, expected_var, atol=tol)

    expected_explained_variance = sorted_eigenvalues[:n_components]
    assert_allclose(incpca.explained_variance_, expected_explained_variance, atol=tol)

    expected_explained_variance_ratio = expected_explained_variance / np.sum(
        sorted_eigenvalues
    )
    assert_allclose(
        incpca.explained_variance_ratio_, expected_explained_variance_ratio, atol=tol
    )

    expected_noise_variance = (
        np.mean(sorted_eigenvalues[n_components:])
        if len(sorted_eigenvalues) > n_components
        else 0.0
    )
    # TODO Fix noise variance computation (It is necessary to update C++ side)
    # assert np.abs(incpca.noise_variance_ - expected_noise_variance) < tol

    expected_transformed_data = centered_data @ components.T
    if whiten:
        scale = np.sqrt(incpca.explained_variance_)
        min_scale = np.finfo(scale.dtype).eps
        scale[scale < min_scale] = np.inf
        expected_transformed_data /= scale

    if not (whiten and n_components == n_samples_seen):
        assert_allclose(_as_numpy(transformed_data), expected_transformed_data, atol=tol)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_on_gold_data(dataframe, queue, whiten, num_blocks, dtype):

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = X.astype(dtype=dtype)
    X_split = np.array_split(X, num_blocks)
    incpca = IncrementalPCA(whiten=whiten)

    for i in range(num_blocks):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        incpca.partial_fit(X_split_df)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    transformed_data = incpca.transform(X_df)
    check_pca_on_gold_data(incpca, dtype, whiten, transformed_data)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_on_gold_data(dataframe, queue, whiten, num_blocks, dtype):

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = X.astype(dtype=dtype)
    incpca = IncrementalPCA(whiten=whiten, batch_size=X.shape[0] // num_blocks)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    incpca.fit(X_df)
    transformed_data = incpca.transform(X_df)

    check_pca_on_gold_data(incpca, dtype, whiten, transformed_data)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_fit_transform_on_gold_data(
    dataframe, queue, whiten, num_blocks, dtype
):

    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = X.astype(dtype=dtype)
    incpca = IncrementalPCA(whiten=whiten, batch_size=X.shape[0] // num_blocks)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    transformed_data = incpca.fit_transform(X_df)

    check_pca_on_gold_data(incpca, dtype, whiten, transformed_data)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("n_components", [None, 1, 5])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 10])
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_partial_fit_on_random_data(
    dataframe, queue, n_components, whiten, num_blocks, row_count, column_count, dtype
):
    seed = 81
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype=dtype)
    X_split = np.array_split(X, num_blocks)
    incpca = IncrementalPCA(n_components=n_components, whiten=whiten)

    for i in range(num_blocks):
        X_split_df = _convert_to_dataframe(
            X_split[i], sycl_queue=queue, target_df=dataframe
        )
        incpca.partial_fit(X_split_df)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    transformed_data = incpca.transform(X_df)
    check_pca(incpca, dtype, whiten, X, transformed_data)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sklearnex_incremental_estimatior_pickle(dataframe, queue, dtype):
    import pickle

    incpca = IncrementalPCA()

    # Check that estimator can be serialized without any data.
    dump = pickle.dumps(incpca)
    incpca_loaded = pickle.loads(dump)

    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(10, 10))
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    X_split_df = _convert_to_dataframe(X_split[0], sycl_queue=queue, target_df=dataframe)
    incpca.partial_fit(X_split_df)
    incpca_loaded.partial_fit(X_split_df)
    dump = pickle.dumps(incpca_loaded)
    incpca_loaded = pickle.loads(dump)
    assert incpca.batch_size == incpca_loaded.batch_size
    assert incpca.n_features_in_ == incpca_loaded.n_features_in_
    assert incpca.n_samples_seen_ == incpca_loaded.n_samples_seen_
    if hasattr(incpca, "_parameter_constraints"):
        assert incpca._parameter_constraints == incpca_loaded._parameter_constraints
    assert incpca.n_jobs == incpca_loaded.n_jobs
    X_split_df = _convert_to_dataframe(X_split[1], sycl_queue=queue, target_df=dataframe)
    incpca.partial_fit(X_split_df)
    incpca_loaded.partial_fit(X_split_df)

    # Check that estimator can be serialized after partial_fit call.
    dump = pickle.dumps(incpca)
    incpca_loaded = pickle.loads(dump)

    assert_allclose(incpca.singular_values_, incpca_loaded.singular_values_, atol=1e-6)
    assert_allclose(incpca.n_samples_seen_, incpca_loaded.n_samples_seen_, atol=1e-6)
    assert_allclose(incpca.n_features_in_, incpca_loaded.n_features_in_, atol=1e-6)
    assert_allclose(incpca.mean_, incpca_loaded.mean_, atol=1e-6)
    assert_allclose(incpca.var_, incpca_loaded.var_, atol=1e-6)
    assert_allclose(
        incpca.explained_variance_, incpca_loaded.explained_variance_, atol=1e-6
    )
    assert_allclose(incpca.components_, incpca_loaded.components_, atol=1e-6)
    assert_allclose(
        incpca.explained_variance_ratio_,
        incpca_loaded.explained_variance_ratio_,
        atol=1e-6,
    )

    # Check that finalized estimator can be serialized.
    dump = pickle.dumps(incpca_loaded)
    incpca_loaded = pickle.loads(dump)

    assert_allclose(incpca.singular_values_, incpca_loaded.singular_values_, atol=1e-6)
    assert_allclose(incpca.n_samples_seen_, incpca_loaded.n_samples_seen_, atol=1e-6)
    assert_allclose(incpca.n_features_in_, incpca_loaded.n_features_in_, atol=1e-6)
    assert_allclose(incpca.mean_, incpca_loaded.mean_, atol=1e-6)
    assert_allclose(incpca.var_, incpca_loaded.var_, atol=1e-6)
    assert_allclose(
        incpca.explained_variance_, incpca_loaded.explained_variance_, atol=1e-6
    )
    assert_allclose(incpca.components_, incpca_loaded.components_, atol=1e-6)
    assert_allclose(
        incpca.explained_variance_ratio_,
        incpca_loaded.explained_variance_ratio_,
        atol=1e-6,
    )

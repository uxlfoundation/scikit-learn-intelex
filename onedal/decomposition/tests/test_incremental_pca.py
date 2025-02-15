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

import numpy as np
import pytest
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import daal_check_version
from onedal.datatypes import from_table
from onedal.decomposition import IncrementalPCA
from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("is_deterministic", [True, False])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_on_gold_data(queue, is_deterministic, whiten, num_blocks, dtype):
    X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    X = X.astype(dtype=dtype)
    X_split = np.array_split(X, num_blocks)
    incpca = IncrementalPCA(is_deterministic=is_deterministic, whiten=whiten)

    for i in range(num_blocks):
        incpca.partial_fit(X_split[i], queue=queue)

    result = incpca.finalize_fit()

    transformed_data = incpca.transform(X, queue=queue)

    expected_n_components_ = 2
    expected_components_ = np.array([[0.83849224, 0.54491354], [-0.54491354, 0.83849224]])
    expected_singular_values_ = np.array([6.30061232, 0.54980396])
    expected_mean_ = np.array([0, 0])
    expected_var_ = np.array([5.6, 2.4])
    expected_explained_variance_ = np.array([7.93954312, 0.06045688])
    expected_explained_variance_ratio_ = np.array([0.99244289, 0.00755711])
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
    if transformed_data.dtype == np.float32:
        tol = 7e-6 if whiten else 1e-6

    assert result.n_components_ == expected_n_components_

    assert_allclose(result.singular_values_, expected_singular_values_, atol=tol)
    assert_allclose(result.mean_, expected_mean_, atol=tol)
    assert_allclose(result.var_, expected_var_, atol=tol)
    assert_allclose(result.explained_variance_, expected_explained_variance_, atol=tol)
    assert_allclose(
        result.explained_variance_ratio_, expected_explained_variance_ratio_, atol=tol
    )
    if is_deterministic and daal_check_version((2024, "P", 500)):
        assert_allclose(result.components_, expected_components_, atol=tol)
        assert_allclose(transformed_data, expected_transformed_data, atol=tol)
    else:
        for i in range(result.n_components_):
            abs_dot_product = np.abs(
                np.dot(result.components_[i], expected_components_[i])
            )
            assert np.abs(abs_dot_product - 1.0) < tol

            if np.dot(result.components_[i], expected_components_[i]) < 0:
                assert_allclose(
                    -transformed_data[i], expected_transformed_data[i], atol=tol
                )
            else:
                assert_allclose(
                    transformed_data[i], expected_transformed_data[i], atol=tol
                )


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("n_components", [None, 1, 5])
@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("num_blocks", [1, 10])
@pytest.mark.parametrize("row_count", [100, 1000])
@pytest.mark.parametrize("column_count", [10, 100])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_on_random_data(
    queue, n_components, whiten, num_blocks, row_count, column_count, dtype
):
    seed = 78
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(row_count, column_count))
    X = X.astype(dtype=dtype)
    X_split = np.array_split(X, num_blocks)
    incpca = IncrementalPCA(n_components=n_components, whiten=whiten)

    for i in range(num_blocks):
        incpca.partial_fit(X_split[i], queue=queue)

    incpca.finalize_fit()

    transformed_data = incpca.transform(X, queue=queue)
    tol = 3e-3 if transformed_data.dtype == np.float32 else 2e-6

    n_components = incpca.n_components_
    expected_n_samples_seen = X.shape[0]
    expected_n_features_in = X.shape[1]
    n_samples_seen = incpca.n_samples_seen_
    n_features_in = incpca.n_features_in_
    assert n_samples_seen == expected_n_samples_seen
    assert n_features_in == expected_n_features_in

    components = incpca.components_
    singular_values = incpca.singular_values_
    centered_data = X - np.mean(X, axis=0)
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

    expected_mean = np.mean(X, axis=0)
    assert_allclose(incpca.mean_, expected_mean, atol=tol)

    expected_var_ = np.var(X, ddof=1, axis=0)
    assert_allclose(incpca.var_, expected_var_, atol=tol)

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

    if daal_check_version((2024, "P", 500)) or not (
        whiten and queue is not None and queue.sycl_device.device_type.name == "gpu"
    ):
        assert_allclose(transformed_data, expected_transformed_data, atol=tol)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_incremental_estimator_pickle(queue, dtype):
    import pickle

    from onedal.decomposition import IncrementalPCA

    incpca = IncrementalPCA()

    # Check that estimator can be serialized without any data.
    dump = pickle.dumps(incpca)
    incpca_loaded = pickle.loads(dump)
    seed = 77
    gen = np.random.default_rng(seed)
    X = gen.uniform(low=-0.3, high=+0.7, size=(10, 10))
    X = X.astype(dtype)
    X_split = np.array_split(X, 2)
    incpca.partial_fit(X_split[0], queue=queue)
    incpca_loaded.partial_fit(X_split[0], queue=queue)
    assert incpca._need_to_finalize == True
    assert incpca_loaded._need_to_finalize == True

    # Check that estimator can be serialized after partial_fit call.
    dump = pickle.dumps(incpca)
    incpca_loaded = pickle.loads(dump)
    assert incpca._need_to_finalize == False
    # Finalize is called during serialization to make sure partial results are finalized correctly.
    assert incpca_loaded._need_to_finalize == False

    partial_n_rows = from_table(incpca._partial_result.partial_n_rows)
    partial_n_rows_loaded = from_table(incpca_loaded._partial_result.partial_n_rows)
    assert_allclose(partial_n_rows, partial_n_rows_loaded)

    partial_crossproduct = from_table(incpca._partial_result.partial_crossproduct)
    partial_crossproduct_loaded = from_table(
        incpca_loaded._partial_result.partial_crossproduct
    )
    assert_allclose(partial_crossproduct, partial_crossproduct_loaded)

    partial_sum = from_table(incpca._partial_result.partial_sum)
    partial_sum_loaded = from_table(incpca_loaded._partial_result.partial_sum)
    assert_allclose(partial_sum, partial_sum_loaded)

    auxiliary_table_count = incpca._partial_result.auxiliary_table_count
    auxiliary_table_count_loaded = incpca_loaded._partial_result.auxiliary_table_count
    assert_allclose(auxiliary_table_count, auxiliary_table_count_loaded)

    for i in range(auxiliary_table_count):
        aux_table = incpca._partial_result.get_auxiliary_table(i)
        aux_table_loaded = incpca_loaded._partial_result.get_auxiliary_table(i)
        assert_allclose(from_table(aux_table), from_table(aux_table_loaded))

    incpca.partial_fit(X_split[1], queue=queue)
    incpca_loaded.partial_fit(X_split[1], queue=queue)
    assert incpca._need_to_finalize == True
    assert incpca_loaded._need_to_finalize == True

    dump = pickle.dumps(incpca_loaded)
    incpca_loaded = pickle.loads(dump)

    assert incpca._need_to_finalize == True
    assert incpca_loaded._need_to_finalize == False

    incpca.finalize_fit()
    incpca_loaded.finalize_fit()

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

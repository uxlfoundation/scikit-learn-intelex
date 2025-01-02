# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose

# Note: n_components must be 2 for now
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


def test_sklearnex_import():
    from sklearnex.manifold import TSNE

    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    tsne = TSNE(n_components=2, perplexity=2.0).fit(X)
    assert "daal4py" in tsne.__module__


from sklearnex.manifold import TSNE


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_tsne_import(dataframe, queue):
    """Test TSNE compatibility with different backends and queues, and validate sklearnex module."""
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    tsne = TSNE(n_components=2, perplexity=2.0).fit(X_df)
    assert "daal4py" in tsne.__module__
    assert hasattr(tsne, "n_components"), "TSNE missing 'n_components' attribute."
    assert tsne.n_components == 2, "TSNE 'n_components' attribute is incorrect."


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_functionality_and_edge_cases(dataframe, queue, dtype):
    """
    TSNE test covering basic functionality and edge cases using get_dataframes_and_queues.
    """
    # Test basic functionality
    X_basic = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=dtype)
    X_basic_df = _convert_to_dataframe(X_basic, sycl_queue=queue, target_df=dataframe)
    tsne_basic = TSNE(n_components=2, perplexity=2.0, random_state=42)
    embedding_basic = tsne_basic.fit_transform(X_basic_df)
    assert embedding_basic.shape == (4, 2)

    # Test with random data
    X_random = np.random.rand(100, 10).astype(dtype)
    X_random_df = _convert_to_dataframe(X_random, sycl_queue=queue, target_df=dataframe)
    tsne_random = TSNE(n_components=2, perplexity=30.0, random_state=42)
    embedding_random = tsne_random.fit_transform(X_random_df)
    assert embedding_random.shape == (100, 2)

    # Test reproducibility
    X_repro = np.random.rand(50, 10).astype(dtype)
    X_repro_df = _convert_to_dataframe(X_repro, sycl_queue=queue, target_df=dataframe)
    tsne_repro_1 = TSNE(n_components=2, random_state=42).fit_transform(X_repro_df)
    tsne_repro_2 = TSNE(n_components=2, random_state=42).fit_transform(X_repro_df)
    tsne_repro_1_np = _as_numpy(tsne_repro_1)
    tsne_repro_2_np = _as_numpy(tsne_repro_2)
    assert_allclose(tsne_repro_1_np, tsne_repro_2_np, rtol=1e-5)

    # Test large data
    X_large = np.random.rand(1000, 50).astype(dtype)
    X_large_df = _convert_to_dataframe(X_large, sycl_queue=queue, target_df=dataframe)
    tsne_large = TSNE(n_components=2, perplexity=50.0, random_state=42)
    embedding_large = tsne_large.fit_transform(X_large_df)
    assert embedding_large.shape == (1000, 2)

    # Test valid minimal data
    X_valid = np.array([[0, 0], [1, 1], [2, 2]], dtype=dtype)
    X_valid_df = _convert_to_dataframe(X_valid, sycl_queue=queue, target_df=dataframe)
    tsne_valid = TSNE(n_components=2, perplexity=2, random_state=42)
    embedding_valid = tsne_valid.fit_transform(X_valid_df)
    assert embedding_valid.shape == (3, 2)

    # Edge case: constant data
    X_constant = np.ones((10, 10), dtype=dtype)
    X_constant_df = _convert_to_dataframe(
        X_constant, sycl_queue=queue, target_df=dataframe
    )
    tsne_constant = TSNE(n_components=2, perplexity=5, random_state=42)
    embedding_constant = tsne_constant.fit(X_constant_df).embedding_
    assert embedding_constant.shape == (10, 2)

    # Edge case: empty data
    X_empty = np.empty((0, 10), dtype=dtype)
    with pytest.raises(ValueError):
        TSNE(n_components=2).fit(
            _convert_to_dataframe(X_empty, sycl_queue=queue, target_df=dataframe)
        )

    # Edge case: data with NaN or infinite values
    X_invalid = np.array([[0, 0], [1, np.nan], [2, np.inf]], dtype=dtype)
    with pytest.raises(ValueError):
        TSNE(n_components=2).fit(
            _convert_to_dataframe(X_invalid, sycl_queue=queue, target_df=dataframe)
        )

    # Edge Case: Sparse-Like High-Dimensional Data
    np.random.seed(42)
    X_sparse_like = np.random.rand(50, 500).astype(dtype) * (
        np.random.rand(50, 500) > 0.99
    )
    X_sparse_like_df = _convert_to_dataframe(
        X_sparse_like, sycl_queue=queue, target_df=dataframe
    )
    try:
        tsne = TSNE(n_components=2, perplexity=30.0)
        tsne.fit(X_sparse_like_df)
    except Exception as e:
        pytest.fail(f"TSNE failed on sparse-like high-dimensional data: {e}")

    # Edge Case: Extremely Low Perplexity
    X_low_perplexity = np.random.rand(10, 5).astype(dtype)
    X_low_perplexity_df = _convert_to_dataframe(
        X_low_perplexity, sycl_queue=queue, target_df=dataframe
    )
    try:
        tsne_low_perplexity = TSNE(n_components=2, perplexity=0.5)
        tsne_low_perplexity.fit(X_low_perplexity_df)
    except Exception as e:
        pytest.fail(f"TSNE failed with low perplexity: {e}")


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_with_specific_complex_dataset(dataframe, queue, dtype):
    """Test TSNE with a specific, highly diverse dataset."""
    complex_array = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [-1e-9, 1e-9, -1e-9, 1e-9],
            [-1e9, 1e9, -1e9, 1e9],
            [1e-3, 1e3, -1e3, -1e-3],
            [0, 1e9, -1e-9, 1],
            [1, -1, 1, -1],
            [42, 42, 42, 42],
            [0, 0, 1, -1],
            [-1e5, 0, 1e5, -1],
            [2e9, 2e-9, -2e9, -2e-9],
            [3, -3, 3e3, -3e-3],
            [5e-5, 5e5, -5e-5, -5e5],
            [1, 0, -1e8, 1e8],
            [9e-7, -9e7, 9e-7, -9e7],
            [4e-4, 4e4, -4e-4, -4e4],
            [6e-6, -6e6, 6e6, -6e-6],
            [8, -8, 8e8, -8e-8],
        ],
        dtype=dtype,
    )

    complex_array_df = _convert_to_dataframe(
        complex_array, sycl_queue=queue, target_df=dataframe
    )

    try:
        tsne = TSNE(n_components=2, perplexity=5.0, random_state=42)
        embedding = tsne.fit_transform(complex_array_df)
        assert embedding.shape == (
            complex_array.shape[0],
            2,
        ), "TSNE embedding shape is incorrect."
    except Exception as e:
        pytest.fail(f"TSNE failed on the specific complex dataset: {e}")

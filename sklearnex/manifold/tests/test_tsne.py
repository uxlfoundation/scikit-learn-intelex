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


def test_basic_tsne_functionality():
    """Test TSNE with valid data: basic functionality, random data, reproducibility, and edge cases."""
    # Test basic functionality
    X_basic = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    tsne = TSNE(n_components=2, perplexity=2.0).fit(X_basic)
    assert tsne.embedding_.shape == (4, 2)

    # Test with random data
    np.random.seed(42)
    X_random = np.random.rand(100, 10)
    tsne_random = TSNE(n_components=2, perplexity=30.0).fit(X_random)
    assert tsne_random.embedding_.shape == (100, 2)

    # Test reproducibility
    X_repro = np.random.rand(50, 10)
    tsne_1 = TSNE(n_components=2, random_state=42).fit_transform(X_repro)
    tsne_2 = TSNE(n_components=2, random_state=42).fit_transform(X_repro)
    assert_allclose(tsne_1, tsne_2, rtol=1e-5)

    # Test perplexity close to dataset size
    X_perplexity = np.random.rand(10, 5)
    tsne_perplexity = TSNE(n_components=2, perplexity=9).fit(X_perplexity)
    assert tsne_perplexity.embedding_.shape == (10, 2)

    # Test large data
    X_large = np.random.rand(1000, 50)
    tsne_large = TSNE(n_components=2, perplexity=50.0).fit(X_large)
    assert tsne_large.embedding_.shape == (1000, 2)

    # Test valid minimal data
    X_valid = np.array([[0, 0], [1, 1], [2, 2]])
    tsne_valid = TSNE(n_components=2, perplexity=2).fit(X_valid)
    assert tsne_valid.embedding_.shape == (3, 2)

    # Edge case: constant data
    X_constant = np.ones((10, 10))
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    embedding = tsne.fit(X_constant).embedding_
    assert embedding.shape == (10, 2), f"Unexpected embedding shape: {embedding.shape}"

    # Edge case: empty data
    X_empty = np.empty((0, 10))
    with pytest.raises(ValueError):
        TSNE(n_components=2).fit(X_empty)

    # Edge case: data with NaN or infinite values
    X_invalid = np.array([[0, 0], [1, np.nan], [2, np.inf]])
    with pytest.raises(ValueError):
        TSNE(n_components=2).fit(X_invalid)

    # Edge Case: Sparse-Like High-Dimensional Data
    np.random.seed(42)
    X_sparse_like = np.random.rand(50, 10000) * (np.random.rand(50, 10000) > 0.99)
    try:
        tsne = TSNE(n_components=2, perplexity=30.0)
        tsne.fit(X_sparse_like)
    except Exception as e:
        pytest.fail(f"TSNE failed on sparse-like high-dimensional data: {e}")

    # Edge Case: Extremely Low Perplexity
    X = np.random.rand(10, 5)
    try:
        tsne_low_perplexity = TSNE(n_components=2, perplexity=0.5)
        tsne_low_perplexity.fit(X)
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


@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues(device_filter_="gpu")
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_gpu_validation(dataframe, queue, dtype):
    """
    GPU validation test for TSNE with a specific complex dataset.
    """
    # Complex dataset for testing
    gpu_validation_array = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [-1e9, 1e9, -1e9, 1e9],
            [1e-3, 1e3, -1e3, -1e-3],
            [1, -1, 1, -1],
            [0, 1e9, -1e-9, 1],
            [-7e11, 7e11, -7e-11, 7e-11],
            [4e-4, 4e4, -4e-4, -4e4],
            [6e-6, -6e6, 6e6, -6e-6],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=dtype,
    )

    expected_shape = (gpu_validation_array.shape[0], 2)
    gpu_array_df = _convert_to_dataframe(
        gpu_validation_array, sycl_queue=queue, target_df=dataframe
    )
    try:
        tsne = TSNE(n_components=2, perplexity=3.0, random_state=42)
        embedding = tsne.fit_transform(gpu_array_df)
        assert (
            embedding.shape == expected_shape
        ), f"Incorrect embedding shape on GPU: {embedding.shape}."
        assert np.all(
            np.isfinite(embedding)
        ), "Embedding contains NaN or infinite values on GPU."
        assert np.any(
            embedding != 0
        ), "GPU embedding contains only zeros, which is invalid."

    except Exception as e:
        pytest.fail(f"TSNE failed on GPU validation test: {e}")

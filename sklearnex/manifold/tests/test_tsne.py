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
from sklearn.metrics.pairwise import pairwise_distances

# Note: n_components must be 2 for now
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import(dataframe, queue):
    """Test TSNE compatibility with different backends and queues, and validate sklearnex module."""
    from sklearnex.manifold import TSNE

    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    tsne = TSNE(n_components=2, perplexity=2.0, random_state=42, init="pca").fit(X_df)
    embedding = tsne.fit_transform(X_df)
    embedding = _as_numpy(embedding)
    assert "daal4py" in tsne.__module__
    assert tsne.n_components == 2
    assert tsne.perplexity == 2.0
    assert tsne.random_state == 42
    assert tsne.init == "pca"


@pytest.mark.parametrize(
    "X_generator,n_components,perplexity,expected_shape,should_raise",
    [
        pytest.param(
            lambda rng: np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
            2,
            2.0,
            (4, 2),
            False,
            id="Basic functionality",
        ),
        pytest.param(
            lambda rng: rng.random((100, 10)),
            2,
            30.0,
            (100, 2),
            False,
            id="Random data",
        ),
        pytest.param(
            lambda rng: np.array([[0, 0], [1, 1], [2, 2]]),
            2,
            2.0,
            (3, 2),
            False,
            id="Valid minimal data",
        ),
        pytest.param(
            lambda rng: np.empty((0, 10)),
            2,
            5.0,
            None,
            True,
            id="Empty data",
        ),
        pytest.param(
            lambda rng: np.array([[0, 0], [1, np.nan], [2, np.inf]]),
            2,
            5.0,
            None,
            True,
            id="Data with NaN/Inf",
        ),
        pytest.param(
            lambda rng: rng.random((50, 500)) * (rng.random((50, 500)) > 0.99),
            2,
            30.0,
            (50, 2),
            False,
            id="Sparse-like high-dimensional data",
        ),
        pytest.param(
            lambda rng: np.hstack(
                [
                    np.ones((50, 1)),  # First column is 1
                    rng.random((50, 499)) * (rng.random((50, 499)) > 0.99),
                ]
            ),
            2,
            30.0,
            (50, 2),
            False,
            id="Sparse-like data with constant column",
        ),
        pytest.param(
            lambda rng: np.where(
                np.arange(50 * 500).reshape(50, 500) % 10 == 0, 0, rng.random((50, 500))
            ),
            2,
            30.0,
            (50, 2),
            False,
            id="Sparse-like data with every tenth element zero",
        ),
        pytest.param(
            lambda rng: rng.random((10, 5)),
            2,
            0.5,
            (10, 2),
            False,
            id="Extremely low perplexity",
        ),
    ],
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_functionality_and_edge_cases(
    X_generator,
    n_components,
    perplexity,
    expected_shape,
    should_raise,
    dataframe,
    queue,
    dtype,
):
    from sklearnex.manifold import TSNE

    rng = np.random.default_rng(
        seed=42
    )  # Use generator to ensure independent dataset per test
    X = X_generator(rng)
    X = X.astype(dtype) if X.size > 0 else X
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    if should_raise:
        with pytest.raises(ValueError):
            TSNE(n_components=n_components, perplexity=perplexity).fit_transform(X_df)
    else:
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        embedding = tsne.fit_transform(X_df)
        embedding = _as_numpy(embedding)
        assert embedding.shape == expected_shape
        assert np.all(np.isfinite(embedding))
        assert np.any(embedding != 0)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("init", ["pca", "random"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_constant_data(init, dataframe, queue, dtype):
    from sklearnex.manifold import TSNE

    X = np.ones((10, 10), dtype=dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    tsne = TSNE(n_components=2, init=init, perplexity=5, random_state=42)
    embedding = tsne.fit_transform(X_df)
    embedding = _as_numpy(embedding)
    assert embedding.shape == (10, 2)
    if init == "pca":
        assert np.isclose(embedding[:, 0].std(), 0, atol=1e-6)  # Constant first dimension
        assert np.allclose(embedding[:, 1], 0, atol=1e-6)  # Zero second dimension
    elif init == "random":
        assert np.all(np.isfinite(embedding))


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_reproducibility(dataframe, queue, dtype):
    from sklearnex.manifold import TSNE

    rng = np.random.default_rng(seed=42)
    X = rng.random((50, 10)).astype(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    tsne_1 = TSNE(n_components=2, random_state=42).fit_transform(X_df)
    tsne_2 = TSNE(n_components=2, random_state=42).fit_transform(X_df)
    # in case of dpctl.tensor.usm_ndarray convert to numpy array
    tsne_1 = _as_numpy(tsne_1)
    tsne_2 = _as_numpy(tsne_2)
    assert_allclose(tsne_1, tsne_2, rtol=1e-5)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_complex_and_gpu_validation(dataframe, queue, dtype):
    from sklearnex.manifold import TSNE

    X = np.array(
        [
            [1, 1, 1, 1],
            [1.1, 1.1, 1.1, 1.1],
            [0.9, 0.9, 0.9, 0.9],
            [2e9, 2e-9, -2e9, -2e-9],
            [5e-5, 5e5, -5e-5, -5e5],
            [9e-7, -9e7, 9e-7, -9e7],
            [1, -1, 1, -1],
            [-1e-9, 1e-9, -1e-9, 1e-9],
            [42, 42, 42, 42],
            [8, -8, 8e8, -8e-8],
            [1e-3, 1e3, -1e3, -1e-3],
            [0, 1e9, -1e-9, 1],
            [0, 0, 1, -1],
            [0, 0, 0, 0],
            [-1e5, 0, 1e5, -1],
            [1, 0, -1e8, 1e8],
        ]
    )
    n_components = 2
    perplexity = 3.0
    expected_shape = (16, 2)

    X = X.astype(dtype)
    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    embedding = tsne.fit_transform(X_df)

    # Validate results
    assert embedding.shape == expected_shape
    embedding = _as_numpy(embedding)
    assert np.all(np.isfinite(embedding))
    assert np.any(embedding != 0)

    # Ensure close points in original space remain close in embedding
    group_a_indices = [0, 1, 2]  # Hardcoded index of similar points
    group_b_indices = [3, 4, 5]  # Hardcoded index of dissimilar points from a
    embedding_distances = pairwise_distances(
        X, metric="euclidean"
    )  # Get an array of distance where [i, j] is distance b/t i and j
    # Check for distance b/t two points in group A < distance of this point and any point in group B
    for i in group_a_indices:
        for j in group_a_indices:
            assert (
                embedding_distances[i, j] < embedding_distances[i, group_b_indices].min()
            ), f"Point {i} in Group A is closer to a point in Group B than to another point in Group A."

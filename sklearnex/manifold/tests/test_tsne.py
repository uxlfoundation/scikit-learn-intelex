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


@pytest.mark.parametrize(
    "description,X_generator,n_components,perplexity,expected_shape,should_raise",
    [
        (
            "Test basic functionality",
            lambda rng: np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),
            2,
            2.0,
            (4, 2),
            False,
        ),
        (
            "Test with random data",
            lambda rng: rng.random((100, 10)),
            2,
            30.0,
            (100, 2),
            False,
        ),
        (
            "Test reproducibility",
            lambda rng: rng.random((50, 10)),
            2,
            5.0,
            (50, 2),
            False,
        ),
        (
            "Test large data",
            lambda rng: rng.random((1000, 50)),
            2,
            50.0,
            (1000, 2),
            False,
        ),
        (
            "Test valid minimal data",
            lambda rng: np.array([[0, 0], [1, 1], [2, 2]]),
            2,
            2.0,
            (3, 2),
            False,
        ),
        (
            "Edge case: constant data",
            lambda rng: np.ones((10, 10)),
            2,
            5.0,
            (10, 2),
            False,
        ),
        (
            "Edge case: empty data",
            lambda rng: np.empty((0, 10)),
            2,
            5.0,
            None,
            True,
        ),
        (
            "Edge case: data with NaN or infinite values",
            lambda rng: np.array([[0, 0], [1, np.nan], [2, np.inf]]),
            2,
            5.0,
            None,
            True,
        ),
        (
            "Edge Case: Sparse-Like High-Dimensional Data",
            lambda rng: rng.random((50, 500)) * (rng.random((50, 500)) > 0.99),
            2,
            30.0,
            (50, 2),
            False,
        ),
        (
            "Edge Case: Extremely Low Perplexity",
            lambda rng: rng.random((10, 5)),
            2,
            0.5,
            (10, 2),
            False,
        ),
    ],
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_functionality_and_edge_cases(
    description,
    X_generator,
    n_components,
    perplexity,
    expected_shape,
    should_raise,
    dataframe,
    queue,
    dtype,
):
    """
    TSNE test covering multiple functionality and edge cases using parameterization.
    """
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
        assert (
            embedding.shape == expected_shape
        ), f"{description}: Incorrect embedding shape."


@pytest.mark.parametrize(
    "description,X,n_components,perplexity,expected_shape,device_filter",
    [
        (
            "Specific complex dataset (CPU/GPU)",
            np.array(
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
                ]
            ),
            2,
            5.0,
            (18, 2),
            "cpu,gpu",
        ),
        (
            "GPU validation dataset",
            np.array(
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
                ]
            ),
            2,
            3.0,
            (11, 2),
            "gpu",
        ),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tsne_complex_and_gpu_validation(
    description, X, n_components, perplexity, expected_shape, device_filter, dtype
):
    """
    TSNE test covering specific complex datasets and GPU validation using parameterization.
    """
    dataframes_and_queues = get_dataframes_and_queues(device_filter_=device_filter)
    for param in dataframes_and_queues:
        dataframe, queue = param.values
        # Convert dataset to specified dtype
        X = X.astype(dtype)
        X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

        try:
            tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
            embedding = tsne.fit_transform(X_df)

            # Validate results
            assert (
                embedding.shape == expected_shape
            ), f"{description}: Incorrect embedding shape."
            if device_filter == "gpu":
                assert np.all(
                    np.isfinite(embedding)
                ), f"{description}: Embedding contains NaN or infinite values."
            assert np.any(
                embedding != 0
            ), f"{description}: Embedding contains only zeros."
        except Exception as e:
            pytest.fail(f"TSNE failed on {description}: {e}")

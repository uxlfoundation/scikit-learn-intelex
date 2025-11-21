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
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.model_selection import train_test_split

from onedal.tests.utils._dataframes_support import _as_numpy

try:
    import dpctl
    from dpctl import SyclQueue
    from mpi4py import MPI

    mpi_libs_available = True
    gpu_is_available = dpctl.has_gpu_devices()
except (ImportError, ModuleNotFoundError):
    mpi_libs_available = False

_mpi_libs_and_gpu_available = mpi_libs_available and gpu_is_available


def _get_local_tensor(full_data):
    """Splits data across ranks.

    Called on each rank to extract the subset of data assigned to that rank.

    Args:
        full_data (numpy or dpnp array): The entire set of data

    Returns:
        local_data (numpy or dpnp array): The subset of data used by the rank
    """

    # create sycl queue and gather communicator details
    q = SyclQueue("gpu")
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # divide data across ranks and move to dpnp array
    data_rows = full_data.shape[0]
    local_start = rank * data_rows // size
    local_end = (1 + rank) * data_rows // size
    local_data = full_data[local_start:local_end]

    return local_data


def _generate_regression_data(n_samples, n_features, dtype=np.float64, random_state=42):
    # Generates regression data and divides between train and test
    X, y = make_regression(
        n_samples=n_samples, n_features=n_features, random_state=random_state
    )
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def _generate_classification_data(
    n_samples, n_features, n_classes=2, dtype=np.float64, random_state=42
):
    # Generates classification data and divides between train and test
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        n_informative=int(0.5 * n_classes + 1),
        random_state=random_state,
    )
    X = X.astype(dtype)
    y = y.astype(dtype)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return X_train, X_test, y_train, y_test


def _generate_statistic_data(
    n_samples, n_features=None, dtype=np.float64, random_state=42
):
    # Generates statistical data
    gen = np.random.default_rng(random_state)
    data = gen.uniform(
        low=-0.3,
        high=+0.7,
        size=(n_samples, n_features) if n_features is not None else (n_samples,),
    ).astype(dtype)
    return data


def _generate_clustering_data(
    n_samples, n_features, centers=None, dtype=np.float64, random_state=42
):
    # Generates clustering data and divides between train and test
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=random_state,
    )
    X = X.astype(dtype)
    X_train, X_test = train_test_split(X, random_state=random_state)
    return X_train, X_test


def _spmd_assert_allclose(spmd_result, batch_result, **kwargs):
    """Calls assert_allclose on spmd and batch results.

    Called on each rank to compare the spmd result specific to that rank and
    subset of batch result that corresponds to that rank.

    Args:
        spmd_result (numpy or dpnp array): The result for the subset of data on the rank the function is called from, computed by the spmd estimator
        batch_result (numpy array): The result for all data, computed by the batch estimator

    Raises:
        AssertionError: If all results are not adequately close.
    """

    # extract chunk from batch result to match with local spmd result
    local_batch_result = _get_local_tensor(batch_result)

    assert_allclose(_as_numpy(spmd_result), _as_numpy(local_batch_result), **kwargs)


def _assert_unordered_allclose(spmd_result, batch_result, localize=False, **kwargs):
    """Checks if rows in spmd and batch results are aligned, even if not in the same order.

    Called to verify correct unordered results are present. Useful to check KMeans centers
    or KNN neighbors, where order does not matter. Sorts inputs to handle unordering. Also
    capable of handling localization.

    Args:
        spmd_result (numpy or dpnp array): Result computed by the spmd estimator
        batch_result (numpy array): Result computed by batch estimator
        localize (bool): Whether of not spmd result is specific to the rank, in which case batch result needs to be localized

    Raises:
        AssertionError: If results do not match.
    """
    np_spmd_result = _as_numpy(spmd_result)

    sorted_spmd_result = np_spmd_result[
        np.argsort(np.linalg.norm(np_spmd_result, axis=1))
    ]
    if localize:
        local_batch_result = _get_local_tensor(batch_result)
        sorted_batch_result = local_batch_result[
            np.argsort(np.linalg.norm(local_batch_result, axis=1))
        ]
    else:
        sorted_batch_result = batch_result[
            np.argsort(np.linalg.norm(batch_result, axis=1))
        ]

    assert_allclose(sorted_spmd_result, sorted_batch_result, **kwargs)


def _assert_kmeans_labels_allclose(
    spmd_labels, batch_labels, spmd_centers, batch_centers, **kwargs
):
    """Checks if labels for spmd and batch results are aligned, even cluster indices don't match.

    Called to verify labels are assigned the same way on spmd and batch. Uses raw labels (which
    may not match) to identify cluster center and ensure results match.

    Args:
        spmd_labels (numpy or dpnp array): The labels for the subset of data on the rank the function is called from, computed by the spmd estimator
        batch_labels (numpy array): The labels for all data, computed by the batch estimator
        spmd_centers (numpy or dpnp array): Centers computed by the spmd estimator
        batch_centers (numpy array): Centers computed by batch estimator

    Raises:
        AssertionError: If clusters are not correctly assigned.
    """

    np_spmd_labels = _as_numpy(spmd_labels)
    np_spmd_centers = _as_numpy(spmd_centers)
    local_batch_labels = _get_local_tensor(batch_labels)
    assert_allclose(
        np_spmd_centers[np_spmd_labels],
        batch_centers[local_batch_labels],
        **kwargs,
    )

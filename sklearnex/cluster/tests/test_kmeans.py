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

from contextlib import nullcontext

import array_api_strict
import numpy as np
import pandas as pd
import polars as pl
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose
from sklearn.datasets import make_blobs

if hasattr(sp, "csr_array"):
    CSR_CTOR = sp.csr_array
else:
    CSR_CTOR = sp.csr_matrix

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
    get_queues,
    torch_available,
    torch_xpu_available,
)
from onedal.tests.utils._device_selection import is_sycl_device_available
from sklearnex import config_context
from sklearnex.cluster import KMeans
from sklearnex.tests.utils import _IS_INTEL

if dpnp_available:
    import dpnp
if torch_available:
    import torch


def generate_dense_dataset(n_samples, n_features, density, n_clusters):
    np.random.seed(2024 + n_samples + n_features + n_clusters)
    X, _ = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=n_clusters,
        cluster_std=1.0,
        random_state=42,
    )
    mask = np.random.binomial(1, density, (n_samples, n_features))
    X = X * mask
    return X


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_sklearnex_import_for_dense_data(dataframe, queue, algorithm, init):
    from sklearnex.cluster import KMeans

    X_dense = generate_dense_dataset(1000, 10, 0.5, 3)
    X_dense_df = _convert_to_dataframe(X_dense, sycl_queue=queue, target_df=dataframe)

    kmeans_dense = KMeans(
        n_clusters=3, random_state=0, algorithm=algorithm, init=init
    ).fit(X_dense_df)

    if daal_check_version((2023, "P", 200)):
        assert "sklearnex" in kmeans_dense.__module__
    else:
        assert "daal4py" in kmeans_dense.__module__


@pytest.mark.skipif(
    not daal_check_version((2024, "P", 700)),
    reason="Sparse data requires oneDAL>=2024.7.0",
)
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
@pytest.mark.parametrize("init", ["k-means++", "random"])
def test_sklearnex_import_for_sparse_data(queue, algorithm, init):
    from sklearnex.cluster import KMeans

    X_dense = generate_dense_dataset(1000, 10, 0.5, 3)
    X_sparse = CSR_CTOR(X_dense)

    kmeans_sparse = KMeans(
        n_clusters=3, random_state=0, algorithm=algorithm, init=init
    ).fit(X_sparse)

    assert "sklearnex" in kmeans_sparse.__module__


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
def test_results_on_dense_gold_data(dataframe, queue, algorithm):
    from sklearnex.cluster import KMeans

    X_train = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    X_test = np.array([[0, 0], [12, 3]])
    X_train_df = _convert_to_dataframe(X_train, sycl_queue=queue, target_df=dataframe)
    X_test_df = _convert_to_dataframe(X_test, sycl_queue=queue, target_df=dataframe)

    kmeans = KMeans(n_clusters=2, random_state=0, algorithm=algorithm).fit(X_train_df)

    if queue and queue.sycl_device.is_gpu:
        # KMeans Init Dense GPU implementation is different from CPU
        expected_cluster_labels = np.array([0, 1], dtype=np.int32)
        expected_cluster_centers = np.array([[1.0, 2.0], [10.0, 2.0]], dtype=np.float32)
        expected_inertia = 16.0
    else:
        expected_cluster_labels = np.array([1, 0], dtype=np.int32)
        expected_cluster_centers = np.array([[10.0, 2.0], [1.0, 2.0]], dtype=np.float32)
        expected_inertia = 16.0

    assert_allclose(expected_cluster_labels, _as_numpy(kmeans.predict(X_test_df)))
    assert_allclose(expected_cluster_centers, _as_numpy(kmeans.cluster_centers_))
    assert expected_inertia == kmeans.inertia_


@pytest.mark.skipif(
    not daal_check_version((2024, "P", 700)),
    reason="Sparse data requires oneDAL>=2024.7.0",
)
@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("init", ["k-means++", "random", "arraylike"])
@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
@pytest.mark.parametrize(
    "dims", [(1000, 10, 0.95, 3), (50000, 100, 0.75, 10), (10000, 10, 0.8, 5)]
)
def test_dense_vs_sparse(queue, init, algorithm, dims):
    from sklearnex.cluster import KMeans

    if init == "random" or (not _IS_INTEL and init == "k-means++"):
        pytest.skip(f"{init} initialization for sparse K-means is non-conformant.")

    # For higher level of sparsity (smaller density) the test may fail
    n_samples, n_features, density, n_clusters = dims
    X_dense = generate_dense_dataset(n_samples, n_features, density, n_clusters)
    X_sparse = CSR_CTOR(X_dense)

    if init == "arraylike":
        np.random.seed(2024 + n_samples + n_features + n_clusters)
        init = X_dense[np.random.choice(n_samples, size=n_clusters, replace=False)]

    kmeans_dense = KMeans(
        n_clusters=n_clusters, random_state=0, init=init, algorithm=algorithm
    ).fit(X_dense)
    kmeans_sparse = KMeans(
        n_clusters=n_clusters, random_state=0, init=init, algorithm=algorithm
    ).fit(X_sparse)

    assert_allclose(
        kmeans_dense.cluster_centers_,
        kmeans_sparse.cluster_centers_,
    )


@pytest.mark.parametrize("output_format", ["set_output", "config_context"])
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
def test_transform_output_torch(output_format, transform_output):
    torch = pytest.importorskip("torch")

    X_np = generate_dense_dataset(200, 10, 0.5, 3)
    X_torch = torch.tensor(X_np, device="cpu")

    with config_context(array_api_dispatch=True):
        km = KMeans(n_clusters=3, random_state=0, n_init=1)
        if output_format == "set_output":
            km.set_output(transform=transform_output)
            km.fit(X_torch)
            result = km.transform(X_torch)
        else:
            with config_context(transform_output=transform_output):
                km.fit(X_torch)
                result = km.transform(X_torch)

    expected_type = pl.DataFrame if transform_output == "polars" else pd.DataFrame
    assert isinstance(result, expected_type)


# Only numpy and dpnp: array_api_strict + polars/pandas fails in sklearn itself.
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("numpy,dpnp"))
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
def test_transform_output_gpu(dataframe, queue, transform_output):
    X_np = generate_dense_dataset(200, 10, 0.5, 3)
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True, transform_output=transform_output):
        km = KMeans(n_clusters=3, random_state=0, n_init=1)
        km.fit(X)
        result = km.transform(X)

    expected_type = pl.DataFrame if transform_output == "polars" else pd.DataFrame
    assert isinstance(result, expected_type)


# Excludes pandas (converted to numpy by validate_data, output type won't match).
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("numpy,dpnp,array_api")
)
def test_array_api_dispatch_output_type(dataframe, queue):
    X_np = generate_dense_dataset(200, 10, 0.5, 3)
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        km = KMeans(n_clusters=3, random_state=0, n_init=1)
        km.fit(X)
        pred = km.predict(X)
        trans = km.transform(X)
        sc = km.score(X)

        assert type(pred) == type(X)
        assert type(trans) == type(X)
        assert type(km.cluster_centers_) == type(X)
        assert isinstance(sc, float)


@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Relies on functionality introduced in later scikit-learn versions.",
)
@pytest.mark.skipif(not dpnp_available, reason="Functionality to test requires DPNP.")
@pytest.mark.skipif(
    not is_sycl_device_available("gpu"), reason="Test for GPU-specific functionality."
)
def test_cov_error_on_incompatible_devices(with_array_api):
    import dpnp

    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(50, 3), dtype=np.float32)
    X_cpu = dpnp.array(X, device="cpu")
    X_gpu = dpnp.array(X, device="gpu")

    err_match = "device|queue"

    model = KMeans(algorithm="lloyd").fit(X_gpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.predict(X_cpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.transform(X_cpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.score(X_cpu)

    model.fit(X_cpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.predict(X_gpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.transform(X_gpu)
    with pytest.raises(ValueError, match=err_match):
        _ = model.score(X_gpu)


@pytest.mark.parametrize("array_api", [False, True])
def test_sparse_predict_on_dense_fit(array_api):
    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(50, 3), dtype=np.float32)
    X_sp = CSR_CTOR(X)

    with config_context(array_api_dispatch=array_api):
        model = KMeans().fit(X)
        sp_pred = model.predict(X_sp)
        sp_transf = model.transform(X_sp)
        sp_score = model.score(X_sp)

        dense_pred = model.predict(X)
        dense_transf = model.transform(X)
        dense_score = model.score(X)

        np.testing.assert_allclose(sp_pred, dense_pred)
        np.testing.assert_allclose(sp_transf, dense_transf, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(sp_score, dense_score, rtol=1e-6, atol=1e-6)


def _convert(arr, xp, device):
    """Convert a numpy array to the array-API backend ``xp`` on ``device``."""
    if xp is np:
        return arr
    return xp.asarray(arr, device=device)


# (xp, device) array-API input combinations, CPU and GPU; device-specific entries
# are dropped at collection time when the hardware/library is unavailable.
# dpnp arrays are SYCL arrays even on "cpu", so they need a SYCL-enabled sklearnex
# build (``_dpc_backend``) to be converted -- a CPU-only build raises "installation
# does not have SYCL support". ``is_sycl_device_available`` is not enough: it uses a
# dpctl queue that succeeds regardless of whether sklearnex was built with DPC.
# TODO: array_api_strict is gated on numpy >= 2.1 only because numpy < 2.1 cannot
# export a readonly array through DLPack, so converting the readonly fitted
# ``components_`` raises BufferError. Remove this guard once the oneDAL data
# conversion handles readonly arrays (tracked separately).
_array_api_inputs = (
    [(np, None)]
    + (
        [(array_api_strict, None)]
        if _package_check_version("2.1.0", np.__version__)
        else []
    )
    + ([(dpnp, "cpu")] if dpnp_available and _dpc_backend is not None else [])
    + (
        [(dpnp, "gpu")]
        if dpnp_available and _dpc_backend is not None and is_sycl_device_available("gpu")
        else []
    )
    + ([(torch, "cpu")] if torch_available else [])
    + ([(torch, "xpu")] if torch_xpu_available else [])
)


def _assert_transform_output_matches_default(km, X, transform_output, method):
    """The polars/pandas transform_output wrapping must preserve the values of
    the default (un-wrapped) output, independent of input type/device. Both ways
    of requesting it -- the ``transform_output`` config and ``set_output`` on the
    estimator -- are checked."""
    default = _as_numpy(getattr(km, method)(X))
    expected_type = pl.DataFrame if transform_output == "polars" else pd.DataFrame

    # 1) global config_context(transform_output=...)
    with config_context(transform_output=transform_output):
        out = getattr(km, method)(X)
    assert isinstance(out, expected_type)
    assert_allclose(out.to_numpy(), default, rtol=1e-5, atol=1e-5)

    # 2) per-estimator set_output(transform=...). Restore the original config
    # afterwards -- set_output("default") would *pin* it and override (1) on a
    # later call with the same estimator.
    original_config = getattr(km, "_sklearn_output_config", {}).copy()
    km.set_output(transform=transform_output)
    try:
        out = getattr(km, method)(X)
        assert isinstance(out, expected_type)
        assert_allclose(out.to_numpy(), default, rtol=1e-5, atol=1e-5)
    finally:
        km._sklearn_output_config = original_config


@pytest.mark.parametrize("xp,device", _array_api_inputs)
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_transform_output_matches_default(
    xp, device, transform_output, method, with_array_api
):
    X_np = generate_dense_dataset(50, 5, 0.5, 3)
    X = _convert(X_np, xp, device)

    km = KMeans(n_clusters=3, random_state=0, n_init=1).fit(X)
    _assert_transform_output_matches_default(km, X, transform_output, method)


@pytest.mark.skipif(not dpnp_available, reason="Functionality to test requires DPNP.")
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("dpnp"))
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_transform_output_dpnp_no_array_api(dataframe, queue, transform_output, method):
    X_np = generate_dense_dataset(50, 5, 0.5, 3)
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    km = KMeans(n_clusters=3, random_state=0, n_init=1).fit(X)
    _assert_transform_output_matches_default(km, X, transform_output, method)


@pytest.mark.skipif(
    not is_sycl_device_available("gpu"), reason="Test for GPU-specific functionality."
)
@pytest.mark.parametrize("transform_output", ["polars", "pandas"])
@pytest.mark.parametrize("method", ["transform", "fit_transform"])
def test_transform_output_target_offload(transform_output, method):
    X_np = generate_dense_dataset(50, 5, 0.5, 3)

    with config_context(target_offload="gpu"):
        km = KMeans(n_clusters=3, random_state=0, n_init=1).fit(X_np)
        _assert_transform_output_matches_default(km, X_np, transform_output, method)


def _check_kmeans_results(km, X, X_np):
    """Verify the results follow the KMeans definition, independent of which
    local optimum the backend converged to: transform gives the distance to
    each fitted centroid, predict is the closest centroid and score is the
    negative sum of squared distances to the assigned centroid."""
    centers = _as_numpy(km.cluster_centers_)
    pred = _as_numpy(km.predict(X))
    trans = _as_numpy(km.transform(X))
    sc = km.score(X)
    assert isinstance(sc, float)

    expected_distances = np.linalg.norm(X_np[:, None, :] - centers[None, :, :], axis=2)
    assert_allclose(trans, expected_distances, rtol=1e-4, atol=1e-4)
    assert_allclose(pred, trans.argmin(axis=1))
    assert_allclose(sc, -np.square(trans.min(axis=1)).sum(), rtol=1e-5)


@pytest.mark.parametrize("xp,device", _array_api_inputs)
def test_array_api_dispatch_results(xp, device, with_array_api):
    X_np = generate_dense_dataset(50, 5, 0.5, 3)
    X = _convert(X_np, xp, device)

    km = KMeans(n_clusters=3, random_state=0, n_init=1).fit(X)
    # predict/transform/cluster_centers_ follow the input type.
    assert type(km.predict(X)) == type(X)
    assert type(km.transform(X)) == type(X)
    assert type(km.cluster_centers_) == type(X)
    _check_kmeans_results(km, X, X_np)

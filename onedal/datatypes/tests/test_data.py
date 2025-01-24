# ==============================================================================
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
# ==============================================================================

import numpy as np
import pytest
import scipy.sparse as sp
from numpy.testing import assert_allclose

from onedal import _backend, _is_dpc_backend
from onedal.datatypes import from_table, to_table
from onedal.utils._dpep_helpers import dpctl_available

if dpctl_available:
    from onedal.datatypes.tests.common import (
        _assert_sua_iface_fields,
        _assert_tensor_attr,
    )

from onedal.primitives import linear_kernel
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from onedal.tests.utils._device_selection import get_queues
from onedal.utils._array_api import _get_sycl_namespace

data_shapes = [
    pytest.param((1000, 100), id="(1000, 100)"),  # 2-D array
    pytest.param((2000, 50), id="(2000, 50)"),  # 2-D array
    pytest.param((50, 1), id="(50, 1)"),  # 2-D array
    pytest.param((1, 50), id="(1, 50)"),  # 2-D array
    pytest.param((50,), id="(50,)"),  # 1-D array
]

unsupported_data_shapes = [
    pytest.param((2, 3, 4), id="(2, 3, 4)"),
    pytest.param((2, 3, 4, 5), id="(2, 3, 4, 5)"),
]

ORDER_DICT = {"F": np.asfortranarray, "C": np.ascontiguousarray}


if _is_dpc_backend:
    from daal4py.sklearn._utils import get_dtype
    from onedal.cluster.dbscan import BaseDBSCAN
    from onedal.common._policy import _get_policy

    class DummyEstimatorWithTableConversions:

        def fit(self, X, y=None):
            sua_iface, xp, _ = _get_sycl_namespace(X)
            policy = _get_policy(X.sycl_queue, None)
            bs_DBSCAN = BaseDBSCAN()
            types = [xp.float32, xp.float64]
            if get_dtype(X) not in types:
                X = xp.astype(X, dtype=xp.float64)
            dtype = get_dtype(X)
            params = bs_DBSCAN._get_onedal_params(dtype)
            X_table = to_table(X)
            # TODO:
            # check other candidates for the dummy base oneDAL func.
            # oneDAL backend func is needed to check result table checks.
            result = _backend.dbscan.clustering.compute(
                policy, params, X_table, to_table(None)
            )
            result_responses_table = result.responses
            result_responses_df = from_table(
                result_responses_table,
                sua_iface=sua_iface,
                sycl_queue=X.sycl_queue,
                xp=xp,
            )
            return X_table, result_responses_table, result_responses_df

else:

    class DummyEstimatorWithTableConversions:
        pass


def _test_input_format_c_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="C")
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_contiguous_numpy(queue, dtype)


def _test_input_format_f_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="F")
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_f_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_f_contiguous_numpy(queue, dtype)


def _test_input_format_c_not_contiguous_numpy(queue, dtype):
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    dummy_data = np.insert(x_default, range(1, x_default.shape[1]), 8, axis=1)
    x_numpy = dummy_data[:, ::2]

    assert_allclose(x_numpy, x_default)

    assert not x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc

    expected = linear_kernel(x_default, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_not_contiguous_numpy(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_not_contiguous_numpy(queue, dtype)


def _test_input_format_c_contiguous_pandas(queue, dtype):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="C")
    assert x_numpy.flags.c_contiguous
    assert not x_numpy.flags.f_contiguous
    assert not x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_c_contiguous_pandas(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_c_contiguous_pandas(queue, dtype)


def _test_input_format_f_contiguous_pandas(queue, dtype):
    pd = pytest.importorskip("pandas")
    rng = np.random.RandomState(0)
    x_default = np.array(5 * rng.random_sample((10, 4)), dtype=dtype)

    x_numpy = np.asanyarray(x_default, dtype=dtype, order="F")
    assert not x_numpy.flags.c_contiguous
    assert x_numpy.flags.f_contiguous
    assert x_numpy.flags.fnc
    x_df = pd.DataFrame(x_numpy)

    expected = linear_kernel(x_df, queue=queue)
    result = linear_kernel(x_numpy, queue=queue)
    assert_allclose(expected, result)


@pytest.mark.parametrize("queue", get_queues())
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_input_format_f_contiguous_pandas(queue, dtype):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Sporadic failures on GPU sycl_queue.")
    _test_input_format_f_contiguous_pandas(queue, dtype)


def _test_conversion_to_table(dtype):
    np.random.seed()
    if dtype in [np.int32, np.int64]:
        x = np.random.randint(0, 10, (15, 3), dtype=dtype)
    else:
        x = np.random.uniform(-2, 2, (18, 6)).astype(dtype)
    x_table = to_table(x)
    x2 = from_table(x_table)
    assert x.dtype == x2.dtype
    assert np.array_equal(x, x2)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
def test_conversion_to_table(dtype):
    _test_conversion_to_table(dtype)


@pytest.mark.skipif(
    not dpctl_available,
    reason="dpctl is required for checks.",
)
@pytest.mark.skipif(
    not _is_dpc_backend,
    reason="__sycl_usm_array_interface__ support requires DPC backend.",
)
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp", "cpu,gpu")
)
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_input_sua_iface_zero_copy(dataframe, queue, order, dtype):
    """Checking that values ​​representing USM allocations `__sycl_usm_array_interface__`
    are preserved during conversion to onedal table.
    """
    rng = np.random.RandomState(0)
    X_np = np.array(5 * rng.random_sample((10, 59)), dtype=dtype)

    X_np = np.asanyarray(X_np, dtype=dtype, order=order)

    X_dp = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    sua_iface, X_dp_namespace, _ = _get_sycl_namespace(X_dp)

    X_table = to_table(X_dp)
    _assert_sua_iface_fields(X_dp, X_table)

    X_dp_from_table = from_table(
        X_table, sycl_queue=queue, sua_iface=sua_iface, xp=X_dp_namespace
    )
    _assert_sua_iface_fields(X_table, X_dp_from_table)
    _assert_tensor_attr(X_dp, X_dp_from_table, order)


@pytest.mark.skipif(
    not dpctl_available,
    reason="dpctl is required for checks.",
)
@pytest.mark.skipif(
    not _is_dpc_backend,
    reason="__sycl_usm_array_interface__ support requires DPC backend.",
)
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp", "cpu,gpu")
)
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("data_shape", data_shapes)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_table_conversions(dataframe, queue, order, data_shape, dtype):
    """Checking that values ​​representing USM allocations `__sycl_usm_array_interface__`
    are preserved during conversion to onedal table and from onedal table to
    sycl usm array dataformat.
    """
    rng = np.random.RandomState(0)
    X = np.array(5 * rng.random_sample(data_shape), dtype=dtype)

    X = ORDER_DICT[order](X)

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    alg = DummyEstimatorWithTableConversions()
    X_table, result_responses_table, result_responses_df = alg.fit(X)

    for obj in [X_table, result_responses_table, result_responses_df, X]:
        assert hasattr(obj, "__sycl_usm_array_interface__"), f"{obj} has no SUA interface"
    _assert_sua_iface_fields(X, X_table)

    # Work around for saving compute-follows-data execution
    # for CPU sycl context requires cause additional memory
    # allocation using the same queue.
    skip_data_0 = True if queue.sycl_device.is_cpu else False
    # Onedal return table's syclobj is empty for CPU inputs.
    skip_syclobj = True if queue.sycl_device.is_cpu else False
    # TODO:
    # investigate why __sycl_usm_array_interface__["data"][1] is changed
    # after conversion from onedal table to sua array.
    # Test is not turned off because of this. Only check is skipped.
    skip_data_1 = True
    _assert_sua_iface_fields(
        result_responses_df,
        result_responses_table,
        skip_data_0=skip_data_0,
        skip_data_1=skip_data_1,
        skip_syclobj=skip_syclobj,
    )
    assert X.sycl_queue == result_responses_df.sycl_queue
    if order == "F":
        assert X.flags.f_contiguous == result_responses_df.flags.f_contiguous
    else:
        assert X.flags.c_contiguous == result_responses_df.flags.c_contiguous
    # 1D output expected to have the same c_contiguous and f_contiguous flag values.
    assert (
        result_responses_df.flags.c_contiguous == result_responses_df.flags.f_contiguous
    )


@pytest.mark.skipif(
    not _is_dpc_backend,
    reason="__sycl_usm_array_interface__ support requires DPC backend.",
)
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp", "cpu,gpu")
)
@pytest.mark.parametrize("data_shape", unsupported_data_shapes)
def test_sua_iface_interop_invalid_shape(dataframe, queue, data_shape):
    X = np.zeros(data_shape)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    sua_iface, _, _ = _get_sycl_namespace(X)

    expected_err_msg = (
        "Unable to convert from SUA interface: only 1D & 2D tensors are allowed"
    )
    with pytest.raises(ValueError, match=expected_err_msg):
        to_table(X)


@pytest.mark.skipif(
    not _is_dpc_backend,
    reason="__sycl_usm_array_interface__ support requires DPC backend.",
)
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp", "cpu,gpu")
)
@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(np.uint16, id=np.dtype(np.uint16).name),
        pytest.param(np.uint32, id=np.dtype(np.uint32).name),
        pytest.param(np.uint64, id=np.dtype(np.uint64).name),
    ],
)
def test_sua_iface_interop_unsupported_dtypes(dataframe, queue, dtype):
    # sua iface interobility supported only for oneDAL supported dtypes
    # for input data: int32, int64, float32, float64.
    # Checking some common dtypes supported by dpctl, dpnp for exception
    # raise.
    X = np.zeros((10, 20), dtype=dtype)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    sua_iface, _, _ = _get_sycl_namespace(X)

    expected_err_msg = "Unable to convert from SUA interface: unknown data type"
    with pytest.raises(ValueError, match=expected_err_msg):
        to_table(X)


@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("numpy,dpctl,dpnp", "cpu,gpu")
)
def test_to_table_non_contiguous_input(dataframe, queue):
    if dataframe in "dpnp,dpctl" and not _is_dpc_backend:
        pytest.skip("__sycl_usm_array_interface__ support requires DPC backend.")
    X, _ = np.mgrid[:10, :10]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    X = X[:, :3]
    sua_iface, _, _ = _get_sycl_namespace(X)
    # X expected to be non-contiguous.
    assert not X.flags.c_contiguous and not X.flags.f_contiguous
    X_t = to_table(X)
    assert X_t and X_t.shape == (10, 3) and X_t.has_data


@pytest.mark.skipif(
    _is_dpc_backend,
    reason="Required check should be done if no DPC backend.",
)
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("dpctl,dpnp", "cpu,gpu")
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sua_iface_interop_if_no_dpc_backend(dataframe, queue, dtype):
    X = np.zeros((10, 20), dtype=dtype)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    sua_iface, _, _ = _get_sycl_namespace(X)

    expected_err_msg = "SYCL usm array conversion to table requires the DPC backend"
    with pytest.raises(RuntimeError, match=expected_err_msg):
        to_table(X)


@pytest.mark.skipif(
    not _is_dpc_backend, reason="Requires DPC backend for dtype conversion"
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_low_precision_gpu_conversion(dtype, sparse):
    # Use a dummy queue as fp32 hardware is not in public testing

    class DummySyclQueue:
        """This class is designed to act like dpctl.SyclQueue
        to force dtype conversion"""

        class DummySyclDevice:
            has_aspect_fp64 = False

        sycl_device = DummySyclDevice()

    queue = DummySyclQueue()

    if sparse:
        X = sp.random(100, 100, format="csr", dtype=dtype)
    else:
        X = np.random.rand(100, 100).astype(dtype)

    if dtype == np.float64:
        with pytest.warns(
            RuntimeWarning,
            match="Data will be converted into float32 from float64 because device does not support it",
        ):
            X_table = to_table(X, queue=queue)
    else:
        X_table = to_table(X, queue=queue)

    assert X_table.dtype == np.float32
    if dtype == np.float32 and not sparse:
        assert_allclose(X, from_table(X_table))


@pytest.mark.parametrize("X", [None, 5, "test", True, [], np.pi, lambda: None])
@pytest.mark.parametrize("queue", get_queues())
def test_non_array(X, queue):
    # Verify that to and from table doesn't raise errors
    # no guarantee is made about type or content
    err_str = ""

    if np.isscalar(X):
        if np.atleast_2d(X).dtype not in [np.float64, np.float32, np.int64, np.int32]:
            err_str = "Found unsupported array type"
    elif not (X is None or isinstance(X, np.ndarray)):
        err_str = r"\[convert_to_table\] Not available input format for convert Python object to onedal table."

    if err_str:
        with pytest.raises(ValueError, match=err_str):
            to_table(X)
    else:
        X_table = to_table(X, queue=queue)
        from_table(X_table)


@pytest.mark.skipif(
    not _is_dpc_backend, reason="Requires DPC backend for dtype conversion"
)
@pytest.mark.parametrize("X", [None, 5, "test", True, [], np.pi, lambda: None])
def test_low_precision_non_array(X):
    # Use a dummy queue as fp32 hardware is not in public testing

    class DummySyclQueue:
        """This class is designed to act like dpctl.SyclQueue
        to force dtype conversion"""

        class DummySyclDevice:
            has_aspect_fp64 = False

        sycl_device = DummySyclDevice()

    queue = DummySyclQueue()
    test_non_array(X, queue)

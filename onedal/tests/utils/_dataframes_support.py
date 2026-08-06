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
import os

import pytest
import scipy.sparse as sp

from sklearnex import config_context, get_config

try:
    import dpnp

    dpnp_available = True
except ImportError:
    dpnp_available = False

torch_xpu_available = False
try:
    import torch

    torch_available = True

    torch_xpu_available = torch.xpu.is_available()
except ImportError:
    torch_available = False

try:
    # This should be lazy imported in the
    # future along with other popular
    # array_api libraries when testing
    # GPU-no-copy.
    import array_api_strict

    array_api_enabled = lambda: get_config()["array_api_dispatch"]
    array_api_enabled()
    array_api_modules = {"array_api": array_api_strict}


except (ImportError, KeyError):
    array_api_enabled = lambda: False
    array_api_modules = {}


import numpy as np
import pandas as pd

from onedal.datatypes._dlpack import dlpack_to_numpy
from onedal.tests.utils._device_selection import get_queues

test_frameworks = os.environ.get(
    "ONEDAL_PYTEST_FRAMEWORKS", "numpy,pandas,dpnp,array_api,torch"
)

# ``move_to`` has a host round-trip fallback for inputs that lack ``__dlpack__``,
# but only for the exceptions it catches; torch signals this case with
# ``AssertionError``, which escapes instead. Namespaces raising a caught
# exception (e.g. array_api_strict) convert such inputs fine. Any host dataframe
# is affected, not just pandas (polars fails identically), so the pandas probe
# below is representative.
# See https://github.com/scikit-learn/scikit-learn/issues/34046.
host_df_to_torch_working = False
if torch_available:
    try:
        # ``move_to`` does not exist before sklearn 1.8, so importing it inside
        # the probe doubles as the version gate.
        from sklearn.utils._array_api import get_namespace_and_device, move_to

        with config_context(array_api_dispatch=True):
            # ``xp`` must be the array-api-wrapped torch namespace, not the
            # ``torch`` module itself, which lacks ``__array_namespace_info__``.
            # The failure is device-independent, so a host tensor suffices.
            xp, _, device = get_namespace_and_device(torch.empty(0))
            _ = move_to(pd.Series([1, 2, 3]), xp=xp, device=device)
        host_df_to_torch_working = True
    except Exception:
        pass


def get_dataframes_and_queues(dataframe_filter_=None, device_filter_="cpu,gpu"):
    """Get supported dataframes for testing.

    This is meant to be used for testing purposes only.
    It is to be used only for pytest test collection.

    Parameters
    ----------
    dataframe_filter_ : str, default=None
        Configure output pytest.params for the certain dataframe formats.
        When it evaluates False, it will default to value of ``test_frameworks``.
    device_filter_ : str, default="cpu,gpu"
        Configure output pytest.params with certain sycl queue for the dataframe,
        where it is applicable.

    Returns
    -------
    list[pytest.param]
        The list of pytest params, included dataframe name (str),
        sycl queue, if applicable for the test case, and test
        case id (str).

    Notes
    -----
        Do not use filters for the test cases disabling. Use `pytest.skip`
        or `pytest.xfail` instead.

    See Also
    --------
    _convert_to_dataframe : Converted input object to certain dataframe format.
    """
    dataframes_and_queues = []

    # filter dataframe_filter_ based on available test frameworks
    if dataframe_filter_:
        dataframe_filter_ = ",".join(
            [i for i in dataframe_filter_.split(",") if i in test_frameworks]
        )
    else:
        dataframe_filter_ = test_frameworks

    if "numpy" in dataframe_filter_:
        dataframes_and_queues.append(pytest.param("numpy", None, id="numpy"))
    if "pandas" in dataframe_filter_:
        dataframes_and_queues.append(pytest.param("pandas", None, id="pandas"))

    def get_df_and_q(dataframe: str):
        df_and_q = []
        for queue in get_queues(device_filter_):
            if queue:
                id = "{}-{}".format(dataframe, queue.id)
                df_and_q.append(pytest.param(dataframe, queue.values[0], id=id))
        return df_and_q

    if dpnp_available and "dpnp" in dataframe_filter_:
        dataframes_and_queues.extend(get_df_and_q("dpnp"))
    if (
        "array_api" in dataframe_filter_
        and "array_api" in array_api_modules
        or array_api_enabled()
    ):
        dataframes_and_queues.append(pytest.param("array_api", None, id="array_api"))
    if torch_available and "torch" in dataframe_filter_:
        dataframes_and_queues.extend(get_df_and_q("torch"))

    return dataframes_and_queues


# Device labels a namespace supports for the mixed-device tests. GPU-capable
# array-API frameworks may hold data on host ("cpu") or device; torch names its
# device "xpu", dpnp names it "gpu".
_NAMESPACE_DEVICES = {}
if torch_xpu_available:
    _NAMESPACE_DEVICES["torch"] = ("xpu", "cpu")
if dpnp_available:
    _NAMESPACE_DEVICES["gpu-dpnp"] = ("gpu", "cpu")


def mixed_device_params(include_pandas_y=False, include_weight=False, x_devices=None):
    """Parameterize the "same namespace, inputs on possibly different devices"
    tests the array-API-dispatch way.

    Under ``array_api_dispatch`` a fit resolves a single namespace across all
    inputs; X, y (and sample weights) from *different* frameworks (e.g. torch +
    dpnp) is not a supported scenario and sklearn rejects it with a "same
    namespace" error, so those combinations are never emitted. Within one
    namespace, inputs may still live on different devices (xpu/cpu, gpu/cpu) --
    that is what these tests exercise. A pandas ``y``/weight is optionally
    allowed since it is namespace-neutral (host targets alongside an array-API X).

    Parameters
    ----------
    include_pandas_y : bool, default=False
        Also emit combinations with a pandas (host) ``y``/weight.
    include_weight : bool, default=False
        Add a sample-weight column; each row becomes
        ``(X_xp, X_device, y_xp, y_device, w_xp, w_device)`` and the weight
        ranges over the same namespace's devices plus ``None`` (no weight).
    x_devices : tuple of str or None, default=None
        Restrict X to these device labels (e.g. ``("cpu",)`` for CPU-only
        estimators). ``None`` uses every device the namespace supports.

    Yields
    ------
    pytest.param
        ``(X_xp, X_device, y_xp, y_device)`` tuples, or with two extra weight
        fields when ``include_weight``.
    """
    for xp, devices in _NAMESPACE_DEVICES.items():
        module = torch if xp == "torch" else dpnp
        x_dev_list = tuple(d for d in (x_devices or devices) if d in devices)
        y_options = [(module, d, f"{xp}-{d}") for d in devices]
        if include_pandas_y:
            y_options.append((pd, None, "pandas"))
        w_options = [(None, None, "no")] + (
            [(module, d, f"{xp}-{d}") for d in devices]
            + ([(pd, None, "pandas")] if include_pandas_y else [])
            if include_weight
            else []
        )
        for x_device in x_dev_list:
            for y_xp, y_device, y_id in y_options:
                if not include_weight:
                    yield pytest.param(
                        module,
                        x_device,
                        y_xp,
                        y_device,
                        id=f"{xp}-{x_device}-X-{y_id}-y",
                    )
                    continue
                for w_xp, w_device, w_id in w_options:
                    yield pytest.param(
                        module,
                        x_device,
                        y_xp,
                        y_device,
                        w_xp,
                        w_device,
                        id=f"{xp}-{x_device}-X-{y_id}-y-{w_id}-w",
                    )


def _as_numpy(obj, *args, **kwargs):
    """Converted input object to numpy.ndarray format."""
    if dpnp_available and isinstance(obj, dpnp.ndarray):
        return obj.asnumpy(*args, **kwargs)
    if torch_available and isinstance(obj, torch.Tensor):
        # ``Tensor.numpy()`` takes no dtype/order/copy args, so apply them after
        # the host transfer to match the other branches' behavior.
        return np.asarray(obj.cpu().detach().numpy(), *args, **kwargs)
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.to_numpy(*args, **kwargs)
    if sp.issparse(obj):
        return obj.toarray(*args, **kwargs)
    try:
        return np.asarray(obj, *args, **kwargs)
    except (TypeError, RuntimeError, ValueError, BufferError):
        # np.asarray can't read a non-CPU device tensor (e.g. torch on xpu);
        # fall back to the library's standard dlpack host converter (which uses
        # np.from_dlpack). array_api libraries that np.asarray already handles
        # never reach this path.
        return dlpack_to_numpy(obj)


def _convert_to_dataframe(obj, sycl_queue=None, target_df=None, *args, **kwargs):
    """Converted input object to certain dataframe format."""
    if target_df is None:
        return obj
    elif target_df == "numpy":
        # Numpy ndarray.
        # `sycl_queue` arg is ignored.
        return np.asarray(obj, *args, **kwargs)
    # Pandas Dataframe
    elif target_df == "pandas":
        if (
            "dtype" in kwargs
            and hasattr(obj, "astype")
            and np.issubdtype(kwargs["dtype"], np.integer)
        ):
            # Pandas float to int not allowed
            obj = obj.astype(kwargs["dtype"])
        if hasattr(obj, "ndim") and obj.ndim == 1:
            return pd.Series(obj, *args, **kwargs)
        else:
            return pd.DataFrame(obj, *args, **kwargs)
    # DPNP ndarray.
    elif target_df == "dpnp":
        return dpnp.asarray(
            obj, usm_type="device", sycl_queue=sycl_queue, *args, **kwargs
        )
    elif target_df in array_api_modules:
        # Array API input other than DPNP ndarray or Numpy ndarray.

        xp = array_api_modules[target_df]
        return xp.asarray(obj)
    elif target_df == "torch":
        if "dtype" in kwargs:
            kwargs["dtype"] = torch.from_numpy(np.empty(0, dtype=kwargs["dtype"])).dtype
        # Mirror the requested sycl_queue's device so torch tensors don't land on
        # xpu for CPU-queue cases (dpnp honors sycl_queue; torch must too).
        is_gpu = sycl_queue is not None and getattr(
            sycl_queue.sycl_device, "is_gpu", False
        )
        if is_gpu and hasattr(torch, "xpu") and torch.xpu.is_available():
            device = "xpu"
        else:
            device = "cpu"
        return torch.as_tensor(obj, device=device, *args, **kwargs)

    raise RuntimeError("Unsupported dataframe conversion")

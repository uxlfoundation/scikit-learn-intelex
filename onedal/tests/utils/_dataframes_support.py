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

from sklearnex import get_config

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

from onedal.tests.utils._device_selection import get_queues

test_frameworks = os.environ.get(
    "ONEDAL_PYTEST_FRAMEWORKS", "numpy,pandas,dpnp,array_api"
)


def get_dataframes_and_queues(
    dataframe_filter_=None, device_filter_="cpu,gpu", dtypes=None
):
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
    dtypes : iterable of numpy dtypes, default=None
        If provided, the returned params include a ``dtype`` dimension.
        Combos are filtered per queue based on SYCL device aspect
        support (``has_aspect_fp16`` / ``has_aspect_fp64``), so
        fp32-only GPUs never receive fp64 parameters. Host entries
        (``queue is None``) yield every requested dtype.

    Returns
    -------
    list[pytest.param]
        When ``dtypes is None`` — legacy tuples of
        ``(dataframe, queue)``. When ``dtypes`` is provided — tuples
        of ``(dataframe, queue, dtype)``.

    Notes
    -----
        Do not use filters for the test cases disabling. Use `pytest.skip`
        or `pytest.xfail` instead.

    See Also
    --------
    _convert_to_dataframe : Converted input object to certain dataframe format.
    """
    # filter dataframe_filter_ based on available test frameworks
    if dataframe_filter_:
        dataframe_filter_ = ",".join(
            [i for i in dataframe_filter_.split(",") if i in test_frameworks]
        )
    else:
        dataframe_filter_ = test_frameworks

    # Collect (dataframe, queue, id_prefix) triples before applying the
    # optional dtype dimension, so host and device entries share a single
    # expansion path.
    base_entries = []
    if "numpy" in dataframe_filter_:
        base_entries.append(("numpy", None, "numpy"))
    if "pandas" in dataframe_filter_:
        base_entries.append(("pandas", None, "pandas"))

    if dpnp_available and "dpnp" in dataframe_filter_:
        for q_param in get_queues(device_filter_):
            if q_param is None:
                continue
            queue = q_param.values[0]
            base_entries.append(("dpnp", queue, f"dpnp-{q_param.id}"))

    if (
        "array_api" in dataframe_filter_
        and "array_api" in array_api_modules
        or array_api_enabled()
    ):
        base_entries.append(("array_api", None, "array_api"))

    if dtypes is None:
        return [pytest.param(df, q, id=i) for df, q, i in base_entries]

    from onedal.tests.utils._device_selection import _queue_supports_dtype

    out = []
    for df, q, prefix in base_entries:
        for dtype in dtypes:
            if not _queue_supports_dtype(q, dtype):
                continue
            dt_name = np.dtype(dtype).name
            out.append(pytest.param(df, q, dtype, id=f"{prefix}-{dt_name}"))
    return out


def _as_numpy(obj, *args, **kwargs):
    """Converted input object to numpy.ndarray format."""
    if dpnp_available and isinstance(obj, dpnp.ndarray):
        return obj.asnumpy(*args, **kwargs)
    if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
        return obj.to_numpy(*args, **kwargs)
    if sp.issparse(obj):
        return obj.toarray(*args, **kwargs)
    return np.asarray(obj, *args, **kwargs)


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

    raise RuntimeError("Unsupported dataframe conversion")

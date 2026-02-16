from contextlib import nullcontext

import numpy as np
import pandas as pd
import polars as pl
import pytest

from sklearnex import config_context
from sklearnex.decomposition import PCA
from sklearnex.ensemble import RandomForestRegressor
from sklearnex.linear_model import LinearRegression

try:
    import dpnp

    has_dpnp = True
except ImportError:
    has_dpnp = False

try:
    import torch

    has_torch = True
except ImportError:
    has_torch = False


# Note: dtypes from torch do not compare with equality
# against dtypes from NumPy. Same with polars. Hence the
# need for this function.
def assert_is_same_dtype(numpy_dtype, dtype) -> bool:
    if numpy_dtype == np.float32:
        assert "float32" in str(dtype)
    elif numpy_dtype == np.float64:
        assert "float64" in str(dtype)


@pytest.mark.parametrize(
    "estimator", [LinearRegression(), RandomForestRegressor(n_estimators=3)]
)
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "castX,castY,use_array_api",
    [
        (np.asarray, np.asarray, False),
        (np.asarray, np.asarray, True),
        (pd.DataFrame, pd.Series, False),
        (pl.DataFrame, pl.Series, False),
    ]
    + ([(dpnp.array, dpnp.array, True)] if has_dpnp else [])
    + ([(torch.from_numpy, torch.from_numpy, True)] if has_torch else []),
)
def test_dtype_is_preserved_supervised(estimator, dtype, castX, castY, use_array_api):
    X = np.arange(100).reshape((25, 4)).astype(dtype)
    y = np.arange(X.shape[0]).astype(dtype)
    X = castX(X)
    y = castY(y)
    ctx = config_context(array_api_dispatch=True) if use_array_api else nullcontext()
    with ctx:
        pred = estimator.fit(X, y).predict(X)
    assert_is_same_dtype(pred.dtype, dtype)


@pytest.mark.parametrize("estimator,method", [(PCA(), "transform")])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize(
    "castX,use_array_api",
    [
        (np.asarray, False),
        (np.asarray, True),
        (pd.DataFrame, False),
        (pl.DataFrame, False),
    ]
    + ([(dpnp.array, True)] if has_dpnp else [])
    + ([(torch.from_numpy, True)] if has_torch else []),
)
def test_dtype_is_preserved_unsupervised(
    estimator,
    method,
    dtype,
    castX,
    use_array_api,
):
    X = np.arange(100).reshape((25, 4)).astype(dtype)
    y = np.arange(X.shape[0]).astype(dtype)
    X = castX(X)
    ctx = config_context(array_api_dispatch=True) if use_array_api else nullcontext()
    with ctx:
        estimator.fit(X, y)
        pred = getattr(estimator, method)(X)
    assert_is_same_dtype(pred.dtype, dtype)

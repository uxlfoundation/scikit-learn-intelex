# ===============================================================================
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
# ===============================================================================

"""Tests for DataFrame-based input types: Pandas and Polars.

Polars DataFrames are supported via scikit-learn's validation layer (which
converts them to numpy when needed).  These tests verify that sklearnex
estimators accept Polars input without errors and produce results numerically
consistent with numpy input.
"""

import numpy as np
import pytest
from sklearn.datasets import load_iris

# ---------------------------------------------------------------------------
# Optional Polars import — skip the whole module if not installed
# ---------------------------------------------------------------------------
polars = pytest.importorskip("polars", reason="polars is not installed")

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _iris_arrays():
    """Return (X, y) as plain numpy arrays (float64)."""
    X, y = load_iris(return_X_y=True)
    return X.astype(np.float64), y.astype(np.float64)


def _to_polars_df(arr):
    """Convert a 2-D numpy array to a polars.DataFrame."""
    return polars.from_numpy(arr)


def _to_polars_series(arr):
    """Convert a 1-D numpy array to a polars.Series."""
    return polars.Series(arr)


# ---------------------------------------------------------------------------
# Parametrized fixtures — estimators under test
# ---------------------------------------------------------------------------

_ESTIMATORS = [
    "LinearRegression",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
]


def _get_estimator(name):
    """Return a sklearnex estimator class by name."""
    import importlib

    for module_path in (
        "sklearnex.linear_model",
        "sklearnex.neighbors",
        "sklearnex.svm",
        "sklearnex.cluster",
    ):
        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, name):
                return getattr(mod, name)
        except ImportError:
            continue
    # Fall back to sklearn
    from sklearn import linear_model, neighbors

    return getattr(linear_model, name, None) or getattr(neighbors, name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("estimator_name", ["LinearRegression"])
def test_polars_dataframe_fit_predict(estimator_name):
    """Polars DataFrame input: fit + predict should match numpy baseline."""
    X_np, y_np = _iris_arrays()
    X_pol = _to_polars_df(X_np)
    y_pol = _to_polars_series(y_np)

    EstClass = _get_estimator(estimator_name)
    assert EstClass is not None, f"Could not find estimator: {estimator_name}"

    # Fit on numpy baseline
    est_np = EstClass()
    est_np.fit(X_np, y_np)
    pred_np = est_np.predict(X_np)

    # Fit on Polars input
    est_pol = EstClass()
    est_pol.fit(X_pol, y_pol)
    pred_pol = est_pol.predict(X_pol)

    np.testing.assert_allclose(
        pred_pol,
        pred_np,
        rtol=1e-5,
        err_msg=f"{estimator_name}: Polars prediction differs from numpy baseline",
    )


@pytest.mark.parametrize("estimator_name", ["KNeighborsClassifier"])
def test_polars_dataframe_classifier(estimator_name):
    """Polars DataFrame input: classifier fit + predict_proba."""
    X_np, y_np = _iris_arrays()
    y_int = y_np.astype(np.int32)
    X_pol = _to_polars_df(X_np)
    y_pol = _to_polars_series(y_int)

    EstClass = _get_estimator(estimator_name)
    assert EstClass is not None

    est_np = EstClass()
    est_np.fit(X_np, y_int)
    pred_np = est_np.predict(X_np)

    est_pol = EstClass()
    est_pol.fit(X_pol, y_pol)
    pred_pol = est_pol.predict(X_pol)

    np.testing.assert_array_equal(
        pred_pol,
        pred_np,
        err_msg=f"{estimator_name}: Polars predictions differ from numpy baseline",
    )


@pytest.mark.parametrize(
    "X_cols, y_col",
    [
        # fewer features — regression still works
        (slice(0, 2), None),
        # all features
        (slice(None), None),
    ],
    ids=["2-features", "all-features"],
)
def test_polars_dataframe_column_subsets(X_cols, y_col):
    """Polars DataFrames with various column subsets pass through without error."""
    X_np, y_np = _iris_arrays()
    X_sub = X_np[:, X_cols]

    X_pol = _to_polars_df(X_sub)
    y_pol = _to_polars_series(y_np)

    from sklearnex.linear_model import LinearRegression

    est = LinearRegression()
    est.fit(X_pol, y_pol)
    preds = est.predict(X_pol)
    assert preds.shape == (X_sub.shape[0],), "Unexpected prediction shape"


def test_polars_dataframe_no_copy_on_convert():
    """Polars -> numpy conversion should not silently drop rows/columns."""
    X_np, _ = _iris_arrays()
    X_pol = _to_polars_df(X_np)

    # Polars .to_numpy() round-trip
    X_roundtrip = X_pol.to_numpy()
    np.testing.assert_array_equal(X_roundtrip, X_np)

# ==============================================================================
# Copyright 2020 Intel Corporation
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

import contextlib
import gc
import itertools
import pickle
import unittest
import warnings
from collections.abc import Callable
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pytest
import treelite
import xgboost as xgb
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

import daal4py as d4p
from daal4py.mb import gbt_convertors
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

try:
    import catboost as cb

    cb_available = True
except ImportError:
    cb_available = False

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import shap

    shap_available = True
except ImportError:
    shap_available = False


shap_required_version = (2024, "P", 1)
shap_api_change_version = (2025, "P", 0)
shap_supported = daal_check_version(shap_required_version)
shap_api_changed = daal_check_version(shap_api_change_version)
shap_not_supported_str = (
    f"SHAP value calculation only supported for version {shap_required_version} or later"
)
shap_unavailable_str = "SHAP Python package not available"
shap_api_change_str = "SHAP calculation requires 2025.0 API"
cb_unavailable_str = "CatBoost not available"

# CatBoost's SHAP value calculation seems to be buggy
# See https://github.com/catboost/catboost/issues/2556
# Disable SHAP tests temporarily until it's next major version
if cb_available:
    catboost_skip_shap = tuple(map(int, cb.__version__.split("."))) < (1, 4, 0)
else:
    catboost_skip_shap = True
catboost_skip_shap_msg = (
    "CatBoost SHAP calculation is buggy. "
    "See https://github.com/catboost/catboost/issues/2556."
)


# Note: models have an attribute telling whether SHAP calculations
# are supported for it or not. When that attribute is 'False', attempts
# to calculate those preditions will throw an exception. This
# forcefully calculates the predictions anyway with the data that the
# object might have, in order to ensure that cases that state they
# do not support SHAP actually do so because the calculations turn
# to NaN and not due to arbitrary reasons.
def force_shap_predict(model, X):
    fptype = "double" if X.dtype == np.float64 else "float"
    if model._is_regression:
        return model._predict_regression(X, fptype, True, False)
    else:
        return model._predict_classification(X, fptype, "computeClassLabels", True, False)


def make_xgb_model(
    objective: str, base_score: "float | None", sklearn_class: bool, empty_trees: bool
) -> "xgb.Booster | xgb.XGBRegressor | xgb.XGBClassifier":
    params_base_score = {"base_score": base_score} if base_score is not None else {}
    min_split_loss = 1e10 if empty_trees else 0.0
    if objective.startswith("binary:"):
        X, y = make_classification(
            n_samples=11,
            n_classes=2,
            n_features=3,
            n_informative=3,
            n_redundant=0,
            random_state=123,
        )
        # Note: this is in order to test '<' vs. '<=' conditions
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if sklearn_class:
            return xgb.XGBClassifier(
                objective=objective,
                base_score=base_score,
                n_estimators=5,
                min_split_loss=min_split_loss,
                max_depth=3,
                random_state=123,
                n_jobs=1,
            ).fit(X, y)
        else:
            return xgb.train(
                dtrain=xgb.DMatrix(X, y),
                num_boost_round=5,
                params={
                    "objective": objective,
                    "min_split_loss": min_split_loss,
                    "max_depth": 3,
                    "seed": 123,
                    "nthread": 1,
                }
                | params_base_score,
            )
    elif objective.startswith("multi:"):
        X, y = make_classification(
            n_samples=10,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=123,
        )
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if sklearn_class:
            return xgb.XGBClassifier(
                objective=objective,
                base_score=base_score,
                n_estimators=5,
                min_split_loss=min_split_loss,
                max_depth=3,
                random_state=123,
                n_jobs=1,
            ).fit(X, y)
        else:
            return xgb.train(
                dtrain=xgb.DMatrix(X, y),
                num_boost_round=5,
                params={
                    "objective": objective,
                    "min_split_loss": min_split_loss,
                    "num_class": 3,
                    "max_depth": 3,
                    "seed": 123,
                    "nthread": 1,
                }
                | params_base_score,
            )
    else:
        X, y = make_regression(n_samples=2, n_features=4, random_state=123)
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if objective == "reg:gamma":
            y = np.exp((y - y.mean()) / y.std())
        elif objective == "reg:logistic":
            y = (y - y.min()) / (y.max() - y.min())
        if sklearn_class:
            return xgb.XGBRegressor(
                objective=objective,
                base_score=base_score,
                n_estimators=5,
                min_split_loss=min_split_loss,
                max_depth=3,
                random_state=123,
                n_jobs=1,
            ).fit(X, y)
        else:
            return xgb.train(
                dtrain=xgb.DMatrix(X, y),
                num_boost_round=5,
                params={
                    "objective": objective,
                    "min_split_loss": min_split_loss,
                    "max_depth": 3,
                    "seed": 123,
                    "nthread": 1,
                }
                | params_base_score,
            )


@pytest.mark.parametrize("objective", ["reg:squarederror", "reg:gamma", "reg:logistic"])
@pytest.mark.parametrize("base_score", [None, 0.0, 0.1, 10.0])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("from_treelite", [False, True])
def test_xgb_regression(
    objective, base_score, sklearn_class, with_nan, dtype, empty_trees, from_treelite
):
    if (
        (objective == "reg:logistic")
        and (base_score is not None)
        and (base_score > 0.0 or base_score < 1.0)
    ):
        pytest.skip("'base_score' not applicable to binary data.")
    if (objective == "reg:gamma") and (base_score is not None) and (base_score == 0.0):
        pytest.skip("'base_score' of zero not applicable to objective 'reg:gamma'.")
    if sklearn_class and from_treelite:
        pytest.skip()

    xgb_model = make_xgb_model(objective, base_score, sklearn_class, empty_trees)
    if from_treelite:
        xgb_model = treelite.frontend.from_xgboost(xgb_model)
    d4p_model = d4p.mb.convert_model(xgb_model)

    if sklearn_class:
        xgb_model = xgb_model.get_booster()

    num_features = (
        xgb_model.num_features() if not from_treelite else xgb_model.num_feature
    )

    assert d4p_model.model_type == "xgboost" if not from_treelite else "treelite"
    assert d4p_model.is_regressor_
    assert d4p_model.n_classes_ == 1
    assert d4p_model.n_features_in_ == num_features

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, num_features), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    if not from_treelite:
        dm_test = xgb.DMatrix(X_test)
        np.testing.assert_allclose(
            d4p_model.predict(X_test),
            xgb_model.predict(dm_test, output_margin=True),
            atol=1e-5,
            rtol=1e-5,
        )
    else:
        np.testing.assert_allclose(
            d4p_model.predict(X_test),
            treelite.gtil.predict(xgb_model, X_test, pred_margin=True).reshape(-1),
            atol=1e-5,
            rtol=1e-5,
        )


@pytest.mark.parametrize("objective", ["reg:squarederror", "reg:gamma", "reg:logistic"])
@pytest.mark.parametrize("base_score", [None, 0.0, 0.1, 10.0])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("from_treelite", [False, True])
def test_xgb_regression_shap(
    objective, base_score, sklearn_class, with_nan, dtype, empty_trees, from_treelite
):
    if (
        (objective == "reg:logistic")
        and (base_score is not None)
        and (base_score > 0.0 or base_score < 1.0)
    ):
        pytest.skip("'base_score' not applicable to binary data.")
    if (objective == "reg:gamma") and (base_score is not None) and (base_score == 0.0):
        pytest.skip("'base_score' of zero not applicable to objective 'reg:gamma'.")
    if sklearn_class and from_treelite:
        pytest.skip()

    xgb_model = make_xgb_model(objective, base_score, sklearn_class, empty_trees)
    d4p_model = d4p.mb.convert_model(
        xgb_model if not from_treelite else treelite.frontend.from_xgboost(xgb_model)
    )

    if sklearn_class:
        xgb_model = xgb_model.get_booster()

    assert d4p_model.model_type == "xgboost" if not from_treelite else "treelite"
    assert d4p_model.is_regressor_
    assert d4p_model.n_classes_ == 1
    assert d4p_model.n_features_in_ == xgb_model.num_features()

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, xgb_model.num_features()), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan
    dm_test = xgb.DMatrix(X_test)

    if shap_supported:
        np.testing.assert_allclose(
            d4p_model.predict(X_test, pred_contribs=True),
            xgb_model.predict(dm_test, pred_contribs=True),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            d4p_model.predict(X_test, pred_interactions=True),
            xgb_model.predict(dm_test, pred_interactions=True),
            atol=1e-5,
            rtol=1e-5,
        )

    elif not shap_api_changed:
        with pytest.raises(NotImplementedError):
            d4p_model.predict(X_test, pred_contribs=True)


@pytest.mark.parametrize("objective", ["binary:logistic", "binary:logitraw"])
@pytest.mark.parametrize("base_score", [None, 0.2, 0.5])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("from_treelite", [False, True])
def test_xgb_binary_classification(
    objective, base_score, sklearn_class, with_nan, dtype, empty_trees, from_treelite
):
    if sklearn_class and from_treelite:
        pytest.skip()
    xgb_model = make_xgb_model(objective, base_score, sklearn_class, empty_trees)
    if from_treelite:
        xgb_model = treelite.frontend.from_xgboost(xgb_model)
    d4p_model = d4p.mb.convert_model(xgb_model)

    if sklearn_class:
        xgb_model = xgb_model.get_booster()

    num_features = (
        xgb_model.num_features() if not from_treelite else xgb_model.num_feature
    )

    assert d4p_model.model_type == "xgboost" if not from_treelite else "treelite"
    assert d4p_model.is_classifier_
    assert d4p_model.n_classes_ == 2
    assert d4p_model.n_features_in_ == num_features

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, num_features), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan
    dm_test = xgb.DMatrix(X_test)

    if from_treelite:
        tl_pred = treelite.gtil.predict(xgb_model, X_test).reshape(-1)
        if objective == "binary:logitraw":
            tl_pred = 1 / (1 + np.exp(-tl_pred))
        np.testing.assert_allclose(
            d4p_model.predict_proba(X_test)[:, 1],
            tl_pred,
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            d4p_model.predict_proba(X_test)[:, 0],
            1.0 - tl_pred,
            atol=1e-5,
            rtol=1e-5,
        )
    elif objective == "binary:logistic":
        np.testing.assert_allclose(
            d4p_model.predict_proba(X_test)[:, 1],
            xgb_model.predict(dm_test),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            d4p_model.predict_proba(X_test)[:, 0],
            1.0 - xgb_model.predict(dm_test),
            atol=1e-5,
            rtol=1e-5,
        )
    elif objective == "binary:logitraw":
        xgb_prob = 1 / (1 + np.exp(-xgb_model.predict(dm_test)))
        np.testing.assert_allclose(
            d4p_model.predict_proba(X_test)[:, 1],
            xgb_prob,
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            d4p_model.predict_proba(X_test)[:, 0],
            1.0 - xgb_prob,
            atol=1e-5,
            rtol=1e-5,
        )

    np.testing.assert_equal(
        d4p_model.predict(X_test),
        np.argmax(d4p_model.predict_proba(X_test), axis=1),
    )


@pytest.mark.parametrize("objective", ["binary:logistic", "binary:logitraw"])
@pytest.mark.parametrize("base_score", [None, 0.2, 0.5])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("from_treelite", [False, True])
def test_xgb_binary_classification_shap(
    objective, base_score, sklearn_class, with_nan, dtype, empty_trees, from_treelite
):
    if sklearn_class and from_treelite:
        pytest.skip()
    xgb_model = make_xgb_model(objective, base_score, sklearn_class, empty_trees)
    d4p_model = d4p.mb.convert_model(
        xgb_model if not from_treelite else treelite.frontend.from_xgboost(xgb_model)
    )

    if sklearn_class:
        xgb_model = xgb_model.get_booster()

    assert d4p_model.model_type == "xgboost" if not from_treelite else "treelite"
    assert d4p_model.is_classifier_
    assert d4p_model.n_classes_ == 2
    assert d4p_model.n_features_in_ == xgb_model.num_features()

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, xgb_model.num_features()), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan
    dm_test = xgb.DMatrix(X_test)

    if shap_supported:
        np.testing.assert_allclose(
            d4p_model.predict(X_test, pred_contribs=True),
            xgb_model.predict(dm_test, pred_contribs=True),
            atol=1e-5,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            d4p_model.predict(X_test, pred_interactions=True),
            xgb_model.predict(dm_test, pred_interactions=True),
            atol=1e-5,
            rtol=1e-5,
        )

    elif not shap_api_changed:
        with pytest.raises(NotImplementedError):
            d4p_model.predict(X_test, pred_contribs=True)


@pytest.mark.parametrize("objective", ["multi:softmax", "multi:softprob"])
@pytest.mark.parametrize("base_score", [None, 0.2, 0.5])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("from_treelite", [False, True])
def test_xgb_multiclass_classification(
    objective, base_score, sklearn_class, with_nan, dtype, empty_trees, from_treelite
):
    if sklearn_class and from_treelite:
        pytest.skip()
    xgb_model = make_xgb_model(objective, base_score, sklearn_class, empty_trees)
    if from_treelite:
        xgb_model = treelite.frontend.from_xgboost(xgb_model)
    d4p_model = d4p.mb.convert_model(xgb_model)

    if sklearn_class:
        xgb_model = xgb_model.get_booster()

    num_features = (
        xgb_model.num_features() if not from_treelite else xgb_model.num_feature
    )

    assert d4p_model.model_type == "xgboost" if not from_treelite else "treelite"
    assert d4p_model.is_classifier_
    assert d4p_model.n_classes_ == 3
    assert d4p_model.n_features_in_ == num_features

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, num_features), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan
    dm_test = xgb.DMatrix(X_test)

    if from_treelite:
        np.testing.assert_allclose(
            d4p_model.predict_proba(X_test),
            treelite.gtil.predict(xgb_model, X_test).squeeze(),
            atol=1e-5,
            rtol=1e-5,
        )
    elif objective == "multi:softprob":
        np.testing.assert_allclose(
            d4p_model.predict_proba(X_test),
            xgb_model.predict(dm_test),
            atol=1e-5,
            rtol=1e-5,
        )
    elif objective == "multi:softmax":
        np.testing.assert_equal(
            d4p_model.predict(X_test),
            xgb_model.predict(dm_test),
        )

    if shap_supported:
        with pytest.raises(TypeError):
            d4p_model.predict(X_test, pred_contribs=True)
        with pytest.raises(TypeError):
            d4p_model.predict(X_test, pred_interactions=True)
    elif not shap_api_changed:
        with pytest.raises(NotImplementedError):
            d4p_model.predict(X_test, pred_contribs=True)


def test_xgb_early_stop():
    X, y = make_classification(
        n_samples=1_500,
        n_features=10,
        n_informative=3,
        n_classes=4,
        random_state=123,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=123
    )
    xgb_model = xgb.train(
        dtrain=xgb.DMatrix(X_train, y_train),
        evals=[(xgb.DMatrix(X_test, y_test), "test")],
        verbose_eval=False,
        num_boost_round=20,
        early_stopping_rounds=5,
        params={
            "objective": "multi:softprob",
            "learning_rate": 0.3,
            "num_class": 4,
            "seed": 123,
            "n_jobs": 1,
        },
    )
    assert xgb_model.best_iteration < 19
    d4p_model = d4p.mb.convert_model(xgb_model)
    np.testing.assert_allclose(
        d4p_model.predict_proba(X_test),
        xgb_model.inplace_predict(
            X_test,
            iteration_range=(0, xgb_model.best_iteration + 1),
        ),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.parametrize("from_treelite", [False, True])
def test_xgb_unsupported(from_treelite):
    X, y = make_regression(n_samples=10, n_features=4, random_state=123)
    xgb_model = xgb.train(
        dtrain=xgb.DMatrix(X, y),
        num_boost_round=5,
        params={
            "objective": "reg:squarederror",
            "booster": "gblinear",
            "seed": 123,
            "nthread": 1,
        },
    )
    with pytest.raises(TypeError):
        d4p.mb.convert_model(xgb_model)

    xgb_model = xgb.train(
        dtrain=xgb.DMatrix(X, y),
        num_boost_round=5,
        params={
            "objective": "reg:squarederror",
            "booster": "dart",
            "max_depth": 3,
            "seed": 123,
            "nthread": 1,
        },
    )
    if not from_treelite:
        with pytest.raises(TypeError):
            d4p.mb.convert_model(xgb_model)
    else:
        # In this case, TreeLite handles the drop logic on their end in a
        # format that is consumable by daal4py.
        tl_model = treelite.frontend.from_xgboost(xgb_model)
        d4p_model = d4p.mb.convert_model(tl_model)
        np.testing.assert_allclose(
            d4p_model.predict(X),
            treelite.gtil.predict(tl_model, X, pred_margin=True).reshape(-1),
            atol=1e-5,
            rtol=1e-5,
        )

    xgb_model = xgb.train(
        dtrain=xgb.DMatrix(X, y),
        num_boost_round=5,
        params={
            "objective": "reg:quantileerror",
            "quantile_alpha": [0.1, 0.9],
            "max_depth": 3,
            "seed": 123,
            "nthread": 1,
        },
    )
    with pytest.raises(TypeError):
        d4p.mb.convert_model(xgb_model)

    X, y = make_regression(n_samples=10, n_features=2, n_targets=2, random_state=123)
    xgb_model = xgb.train(
        dtrain=xgb.DMatrix(X, y),
        num_boost_round=5,
        params={
            "max_depth": 3,
            "seed": 123,
            "nthread": 1,
        },
    )
    with pytest.raises(TypeError):
        d4p.mb.convert_model(
            xgb_model if not from_treelite else treelite.frontend.from_xgboost(xgb_model)
        )

    X = X.astype(int)
    X -= X.min(axis=0, keepdims=True)
    xgb_model = xgb.train(
        dtrain=xgb.DMatrix(X, y[:, 0], feature_types=["c"] * X.shape[1]),
        num_boost_round=5,
        params={
            "max_depth": 3,
            "seed": 123,
            "nthread": 1,
        },
    )
    with pytest.raises(TypeError):
        d4p.mb.convert_model(
            xgb_model if not from_treelite else treelite.frontend.from_xgboost(xgb_model)
        )


def make_lgb_model(
    objective: str,
    sklearn_class: bool,
    with_nan: bool,
    empty_trees: bool,
    boost_from_average: "None | bool" = None,
) -> "lgb.Booster | lgb.LGBMRegressor | lgb.LGBMClassifier":
    min_data_in_leaf = 5_000 if empty_trees else 2
    params_boost_from_average = (
        {"boost_from_average": boost_from_average}
        if boost_from_average is not None
        else {}
    )
    if objective == "binary":
        X, y = make_classification(
            n_samples=11,
            n_classes=2,
            n_features=3,
            n_informative=3,
            n_redundant=0,
            random_state=123,
        )
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if with_nan:
            X[-1, :] = np.nan
        if sklearn_class:
            return lgb.LGBMClassifier(
                objective=objective,
                n_estimators=5,
                max_depth=3,
                num_leaves=6,
                min_data_in_leaf=min_data_in_leaf,
                min_data_in_bin=1,
                min_sum_hessian_in_leaf=1e-10,
                random_state=123,
                deterministic=True,
                verbose=-1,
                n_jobs=1,
                **params_boost_from_average,
            ).fit(X, y)
        else:
            return lgb.train(
                train_set=lgb.Dataset(X, y),
                num_boost_round=5,
                params={
                    "objective": objective,
                    "max_depth": 3,
                    "num_leaves": 6,
                    "min_data_in_leaf": min_data_in_leaf,
                    "min_data_in_bin": 1,
                    "min_sum_hessian_in_leaf": 1e-10,
                    "num_threads": 1,
                    "verbose": -1,
                    "seed": 123,
                    "deterministic": True,
                }
                | params_boost_from_average,
            )
    elif objective == "multiclass":
        X, y = make_classification(
            n_samples=10,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=123,
        )
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if with_nan:
            X[-1, :] = np.nan
        if sklearn_class:
            return lgb.LGBMClassifier(
                objective=objective,
                n_estimators=5,
                max_depth=3,
                num_leaves=6,
                min_data_in_leaf=min_data_in_leaf,
                min_data_in_bin=1,
                min_sum_hessian_in_leaf=1e-10,
                random_state=123,
                deterministic=True,
                verbose=-1,
                n_jobs=1,
                **params_boost_from_average,
            ).fit(X, y)
        else:
            return lgb.train(
                train_set=lgb.Dataset(X, y),
                num_boost_round=5,
                params={
                    "objective": objective,
                    "num_class": 3,
                    "max_depth": 3,
                    "num_leaves": 6,
                    "min_data_in_leaf": min_data_in_leaf,
                    "min_data_in_bin": 1,
                    "min_sum_hessian_in_leaf": 1e-10,
                    "num_threads": 1,
                    "verbose": -1,
                    "seed": 123,
                    "deterministic": True,
                }
                | params_boost_from_average,
            )
    else:
        X, y = make_regression(n_samples=10, n_features=4, random_state=123)
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if with_nan:
            X[-1, :] = np.nan
        if objective == "gamma":
            y = np.exp((y - y.mean()) / y.std())
        if sklearn_class:
            return lgb.LGBMRegressor(
                objective=objective,
                n_estimators=5,
                max_depth=3,
                num_leaves=6,
                min_data_in_leaf=min_data_in_leaf,
                min_data_in_bin=1,
                min_sum_hessian_in_leaf=1e-10,
                random_state=123,
                deterministic=True,
                verbose=-1,
                n_jobs=1,
                **params_boost_from_average,
            ).fit(X, y)
        else:
            return lgb.train(
                train_set=lgb.Dataset(X, y),
                num_boost_round=5,
                params={
                    "objective": objective,
                    "max_depth": 3,
                    "num_leaves": 6,
                    "min_data_in_leaf": min_data_in_leaf,
                    "min_data_in_bin": 1,
                    "min_sum_hessian_in_leaf": 1e-10,
                    "num_threads": 1,
                    "verbose": -1,
                    "seed": 123,
                    "deterministic": True,
                }
                | params_boost_from_average,
            )


@pytest.mark.parametrize("objective", ["regression", "gamma"])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("boost_from_average", [False, True])
def test_lgb_regression(
    objective, sklearn_class, with_nan, dtype, empty_trees, boost_from_average
):
    lgb_model = make_lgb_model(
        objective, sklearn_class, with_nan, empty_trees, boost_from_average
    )
    d4p_model = d4p.mb.convert_model(lgb_model)

    if sklearn_class:
        lgb_model = lgb_model.booster_

    assert d4p_model.model_type == "lightgbm"
    assert d4p_model.is_regressor_
    assert d4p_model.n_classes_ == 1
    assert d4p_model.n_features_in_ == lgb_model.num_feature()

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, lgb_model.num_feature()), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    np.testing.assert_allclose(
        d4p_model.predict(X_test),
        lgb_model.predict(X_test, raw_score=True),
        atol=1e-5,
        rtol=1e-5,
    )

    if shap_supported:
        np.testing.assert_allclose(
            d4p_model.predict(X_test, pred_contribs=True),
            lgb_model.predict(X_test, pred_contrib=True),
            atol=1e-5,
            rtol=1e-5,
        )

    elif not shap_api_changed:
        with pytest.raises(NotImplementedError):
            d4p_model.predict(X_test, pred_contribs=True)


@pytest.mark.skipif(not shap_available, reason=shap_unavailable_str)
@pytest.mark.parametrize("objective", ["regression", "gamma"])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("boost_from_average", [False, True])
def test_lgb_regression_interactions(
    objective, sklearn_class, with_nan, dtype, empty_trees, boost_from_average
):
    if empty_trees:
        pytest.skip("Case not supported by library 'shap'.")
    lgb_model = make_lgb_model(
        objective, sklearn_class, with_nan, empty_trees, boost_from_average
    )
    d4p_model = d4p.mb.convert_model(lgb_model)

    if sklearn_class:
        lgb_model = lgb_model.booster_

    assert d4p_model.model_type == "lightgbm"
    assert d4p_model.is_regressor_
    assert d4p_model.n_classes_ == 1
    assert d4p_model.n_features_in_ == lgb_model.num_feature()

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, lgb_model.num_feature()), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    if shap_supported:
        # SHAP Python package drops bias terms from the returned matrix, therefore we drop the final row & column
        np.testing.assert_allclose(
            d4p_model.predict(X_test, pred_interactions=True)[:, :-1, :-1],
            shap.TreeExplainer(lgb_model).shap_interaction_values(X_test),
            atol=1e-5,
            rtol=1e-5,
        )

    elif not shap_api_changed:
        with pytest.raises(NotImplementedError):
            d4p_model.predict(X_test, pred_contribs=True)


@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("boost_from_average", [False, True])
def test_lgb_binary_classification(
    sklearn_class, with_nan, dtype, empty_trees, boost_from_average
):
    lgb_model = make_lgb_model(
        "binary", sklearn_class, with_nan, empty_trees, boost_from_average
    )
    d4p_model = d4p.mb.convert_model(lgb_model)

    if sklearn_class:
        lgb_model = lgb_model.booster_

    assert d4p_model.model_type == "lightgbm"
    assert d4p_model.is_classifier_
    assert d4p_model.n_classes_ == 2
    assert d4p_model.n_features_in_ == lgb_model.num_feature()

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, lgb_model.num_feature()), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    np.testing.assert_allclose(
        d4p_model.predict_proba(X_test)[:, 1],
        lgb_model.predict(X_test),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        d4p_model.predict_proba(X_test)[:, 0],
        1.0 - lgb_model.predict(X_test),
        atol=1e-5,
        rtol=1e-5,
    )

    np.testing.assert_equal(
        d4p_model.predict(X_test),
        np.argmax(d4p_model.predict_proba(X_test), axis=1),
    )

    if shap_supported:
        np.testing.assert_allclose(
            d4p_model.predict(X_test, pred_contribs=True),
            lgb_model.predict(X_test, pred_contrib=True),
            atol=1e-5,
            rtol=1e-5,
        )

    elif not shap_api_changed:
        with pytest.raises(NotImplementedError):
            d4p_model.predict(X_test, pred_contribs=True)


@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("with_nan", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("empty_trees", [False, True])
@pytest.mark.parametrize("boost_from_average", [False, True])
def test_lgb_multiclass_classification(
    sklearn_class, with_nan, dtype, empty_trees, boost_from_average
):
    lgb_model = make_lgb_model(
        "multiclass", sklearn_class, with_nan, empty_trees, boost_from_average
    )
    d4p_model = d4p.mb.convert_model(lgb_model)

    if sklearn_class:
        lgb_model = lgb_model.booster_

    assert d4p_model.model_type == "lightgbm"
    assert d4p_model.is_classifier_
    assert d4p_model.n_classes_ == 3
    assert d4p_model.n_features_in_ == lgb_model.num_feature()

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, lgb_model.num_feature()), dtype=dtype)
    if with_nan:
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    np.testing.assert_allclose(
        d4p_model.predict_proba(X_test),
        lgb_model.predict(X_test),
        atol=1e-5,
        rtol=1e-5,
    )

    np.testing.assert_equal(
        d4p_model.predict(X_test),
        np.argmax(d4p_model.predict_proba(X_test), axis=1),
    )


def test_lgb_early_stop():
    X, y = make_classification(
        n_samples=1_500,
        n_features=10,
        n_informative=3,
        n_classes=4,
        random_state=123,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=123
    )
    lgb_model = lgb.train(
        train_set=lgb.Dataset(X_train, y_train),
        valid_sets=[lgb.Dataset(X_test, y_test)],
        num_boost_round=25,
        params={
            "early_stopping_round": 2,
            "early_stopping_min_delta": 0.05,
            "objective": "binary",
            "max_depth": 3,
            "num_leaves": 6,
            "min_data_in_leaf": 2,
            "min_data_in_bin": 1,
            "min_sum_hessian_in_leaf": 1e-10,
            "num_threads": 1,
            "verbose": -1,
            "seed": 123,
            "deterministic": True,
        },
    )
    assert lgb_model.num_trees() < 25
    assert lgb_model.num_trees() > 1
    d4p_model = d4p.mb.convert_model(lgb_model)
    np.testing.assert_almost_equal(
        d4p_model.predict_proba(X_test)[:, 1],
        lgb_model.predict(X_test),
    )


def test_lgb_unsupported():
    X, y = make_regression(n_samples=10, n_features=4, random_state=123)
    lgb_model = lgb.train(
        train_set=lgb.Dataset(X, y),
        num_boost_round=5,
        params={
            "linear_tree": True,
            "max_depth": 3,
            "num_leaves": 6,
            "min_data_in_leaf": 2,
            "min_data_in_bin": 1,
            "min_sum_hessian_in_leaf": 1e-10,
            "num_threads": 1,
            "verbose": -1,
            "seed": 123,
            "deterministic": True,
        },
    )
    with pytest.raises(TypeError):
        d4p.mb.convert_model(lgb_model)

    lgb_model = lgb.train(
        train_set=lgb.Dataset(X, y),
        num_boost_round=5,
        params={
            "boosting": "dart",
            "learning_rate": 0.5,
            "max_depth": 3,
            "num_leaves": 6,
            "min_data_in_leaf": 2,
            "min_data_in_bin": 1,
            "min_sum_hessian_in_leaf": 1e-10,
            "num_threads": 1,
            "verbose": -1,
            "seed": 123,
            "deterministic": True,
        },
    )
    with pytest.raises(TypeError):
        d4p.mb.convert_model(lgb_model)

    lgb_model = lgb.train(
        train_set=lgb.Dataset(X, y),
        num_boost_round=5,
        params={
            "boosting": "rf",
            "bagging_fraction": 0.5,
            "feature_fraction": 0.5,
            "max_depth": 3,
            "num_leaves": 6,
            "min_data_in_leaf": 2,
            "min_data_in_bin": 1,
            "min_sum_hessian_in_leaf": 1e-10,
            "num_threads": 1,
            "verbose": -1,
            "seed": 123,
            "deterministic": True,
        },
    )
    with pytest.raises(TypeError):
        d4p.mb.convert_model(lgb_model)

    X = X.astype(int)
    X -= X.min(axis=0, keepdims=True)
    lgb_model = lgb.train(
        train_set=lgb.Dataset(X, y, categorical_feature=np.arange(X.shape[1]).tolist()),
        num_boost_round=5,
        params={
            "max_depth": 3,
            "num_leaves": 6,
            "min_data_in_leaf": 2,
            "min_data_in_bin": 1,
            "min_sum_hessian_in_leaf": 1e-10,
            "num_threads": 1,
            "verbose": -1,
            "seed": 123,
            "deterministic": True,
        },
    )


def make_cb_model(
    objective: str,
    grow_policy: str,
    nan_mode: str,
    sklearn_class: bool,
    empty_trees: bool = False,
    boost_from_average: "None | bool" = None,
) -> "cb.CatBoostRegressor | cb.CatBoostClassifier | cb.CatBoost":
    with_nan = nan_mode != "Forbidden"
    depth = 0 if empty_trees else 3
    params_boost_from_average = (
        {"boost_from_average": boost_from_average}
        if boost_from_average is not None
        else {}
    )
    if objective == "Logloss":
        X, y = make_classification(
            n_samples=11,
            n_classes=2,
            n_features=3,
            n_informative=3,
            n_redundant=0,
            random_state=123,
        )
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if with_nan:
            X[-1, :] = np.nan
        if sklearn_class:
            return cb.CatBoostClassifier(
                objective=objective,
                grow_policy=grow_policy,
                nan_mode=nan_mode,
                boost_from_average=boost_from_average,
                depth=depth,
                iterations=2,
                random_seed=123,
                thread_count=1,
                save_snapshot=False,
                verbose=0,
                allow_writing_files=False,
            ).fit(X, y)
        else:
            return cb.train(
                pool=cb.Pool(X, y),
                params={
                    "objective": objective,
                    "grow_policy": grow_policy,
                    "nan_mode": nan_mode,
                    "depth": depth,
                    "random_seed": 123,
                    "thread_count": 1,
                    "allow_writing_files": False,
                }
                | params_boost_from_average,
                num_boost_round=2,
                verbose=0,
                save_snapshot=False,
            )

    elif objective == "MultiClass":
        X, y = make_classification(
            n_samples=10,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=123,
        )
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if with_nan:
            X[-1, :] = np.nan
        if sklearn_class:
            return cb.CatBoostClassifier(
                objective=objective,
                grow_policy=grow_policy,
                nan_mode=nan_mode,
                boost_from_average=boost_from_average,
                depth=depth,
                iterations=2,
                random_seed=123,
                thread_count=1,
                save_snapshot=False,
                verbose=0,
                allow_writing_files=False,
            ).fit(X, y)
        else:
            return cb.train(
                pool=cb.Pool(X, y),
                params={
                    "objective": objective,
                    "grow_policy": grow_policy,
                    "nan_mode": nan_mode,
                    "depth": depth,
                    "random_seed": 123,
                    "thread_count": 1,
                    "allow_writing_files": False,
                }
                | params_boost_from_average,
                num_boost_round=2,
                verbose=0,
                save_snapshot=False,
            )

    else:
        X, y = make_regression(n_samples=25, n_features=4, random_state=123)
        X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
        if "Tweedie" in objective:
            y = np.exp((y - y.mean()) / y.std())
        if with_nan:
            X[-1, :] = np.nan
        if sklearn_class:
            return cb.CatBoostRegressor(
                objective=objective,
                grow_policy=grow_policy,
                nan_mode=nan_mode,
                boost_from_average=boost_from_average,
                depth=depth,
                iterations=2,
                random_seed=123,
                thread_count=1,
                save_snapshot=False,
                verbose=0,
                allow_writing_files=False,
            ).fit(X, y)
        else:
            return cb.train(
                pool=cb.Pool(X, y),
                params={
                    "objective": objective,
                    "grow_policy": grow_policy,
                    "nan_mode": nan_mode,
                    "depth": depth,
                    "random_seed": 123,
                    "thread_count": 1,
                    "allow_writing_files": False,
                }
                | params_boost_from_average,
                num_boost_round=2,
                verbose=0,
                save_snapshot=False,
            )


@pytest.mark.skipif(not cb_available, reason=cb_unavailable_str)
@pytest.mark.parametrize("objective", ["RMSE", "Tweedie:variance_power=1.99"])
@pytest.mark.parametrize("boost_from_average", [False, True])
@pytest.mark.parametrize("grow_policy", ["SymmetricTree", "Lossguide"])
@pytest.mark.parametrize("nan_mode", ["Forbidden", "Min", "Max"])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("empty_trees", [False, True])
def test_catboost_regression(
    objective,
    boost_from_average,
    grow_policy,
    nan_mode,
    sklearn_class,
    dtype,
    scale,
    empty_trees,
):
    if boost_from_average and objective != "RMSE":
        pytest.skip("Not implemented in catboost.")
    cb_model = make_cb_model(
        objective, grow_policy, nan_mode, sklearn_class, empty_trees, boost_from_average
    )
    d4p.mb.convert_model(cb_model)
    if scale != 1:
        bias = cb_model.get_scale_and_bias()[1]
        cb_model.set_scale_and_bias(scale, bias)
    d4p_model = d4p.mb.convert_model(cb_model)

    assert d4p_model.model_type == "catboost"
    assert d4p_model.is_regressor_
    assert d4p_model.n_classes_ == 1
    assert d4p_model.n_features_in_ == cb_model.n_features_in_

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, cb_model.n_features_in_), dtype=dtype)
    if nan_mode != "Forbidden":
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    np.testing.assert_allclose(
        d4p_model.predict(X_test),
        cb_model.predict(X_test, prediction_type="RawFormulaVal"),
        atol=1e-5,
        rtol=1e-5,
    )

    if shap_supported:
        shap_pred = force_shap_predict(d4p_model, X_test)
        if d4p_model.supports_shap_:
            assert not np.isnan(shap_pred).any()
        else:
            assert np.isnan(shap_pred).any()


@pytest.mark.skipif(not cb_available, reason=cb_unavailable_str)
@pytest.mark.skipif(catboost_skip_shap, reason=catboost_skip_shap_msg)
@pytest.mark.skipif(not shap_supported, reason=shap_not_supported_str)
@pytest.mark.parametrize("objective", ["RMSE", "Tweedie:variance_power=1.99"])
@pytest.mark.parametrize("boost_from_average", [False, True])
@pytest.mark.parametrize("nan_mode", ["Forbidden", "Min", "Max"])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("empty_trees", [False, True])
def test_catboost_shap(
    objective, boost_from_average, nan_mode, sklearn_class, dtype, scale, empty_trees
):
    if boost_from_average and objective != "RMSE":
        pytest.skip("Not implemented in catboost.")
    cb_model = make_cb_model(
        objective,
        "SymmetricTree",
        nan_mode,
        sklearn_class,
        empty_trees,
        boost_from_average,
    )
    if scale != 1:
        bias = cb_model.get_scale_and_bias()[1]
        cb_model.set_scale_and_bias(scale, bias)
    d4p_model = d4p.mb.convert_model(cb_model)

    if not d4p_model.supports_shap_:
        pytest.skip("Not implemented.")

    assert d4p_model.model_type == "catboost"
    assert d4p_model.is_regressor_
    assert d4p_model.n_classes_ == 1
    assert d4p_model.n_features_in_ == cb_model.n_features_in_

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, cb_model.n_features_in_), dtype=dtype)
    if nan_mode != "Forbidden":
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    np.testing.assert_allclose(
        d4p_model.predict(X_test, pred_contribs=True),
        cb_model.get_feature_importance(cb.Pool(X_test), type="ShapValues"),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.skipif(not cb_available, reason=cb_unavailable_str)
@pytest.mark.parametrize("grow_policy", ["SymmetricTree", "Lossguide"])
@pytest.mark.parametrize("nan_mode", ["Forbidden", "Min", "Max"])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("empty_trees", [False, True])
def test_catboost_binary_classification(
    grow_policy, nan_mode, sklearn_class, dtype, scale, empty_trees
):
    cb_model = make_cb_model("Logloss", grow_policy, nan_mode, sklearn_class, empty_trees)
    if scale != 1:
        bias = cb_model.get_scale_and_bias()[1]
        cb_model.set_scale_and_bias(scale, bias)
    d4p_model = d4p.mb.convert_model(cb_model)

    assert d4p_model.model_type == "catboost"
    assert d4p_model.is_classifier_
    assert d4p_model.n_classes_ == 2
    assert d4p_model.n_features_in_ == cb_model.n_features_in_

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, cb_model.n_features_in_), dtype=dtype)
    if nan_mode != "Forbidden":
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    np.testing.assert_allclose(
        d4p_model.predict_proba(X_test),
        cb_model.predict(X_test, prediction_type="Probability"),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        d4p_model.predict(X_test),
        cb_model.predict(X_test, prediction_type="Class"),
        atol=1e-5,
        rtol=1e-5,
    )

    if shap_supported:
        shap_pred = force_shap_predict(d4p_model, X_test)
        if d4p_model.supports_shap_:
            assert not np.isnan(shap_pred).any()
        else:
            assert np.isnan(shap_pred).any()


@pytest.mark.skipif(not cb_available, reason=cb_unavailable_str)
@pytest.mark.parametrize("grow_policy", ["SymmetricTree", "Lossguide"])
@pytest.mark.parametrize("nan_mode", ["Forbidden", "Min", "Max"])
@pytest.mark.parametrize("sklearn_class", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1.0, 2.5])
@pytest.mark.parametrize("set_bias", [False, True])
@pytest.mark.parametrize("empty_trees", [False, True])
def test_catboost_multiclass_classification(
    grow_policy, nan_mode, sklearn_class, dtype, scale, set_bias, empty_trees
):
    cb_model = make_cb_model(
        "MultiClass", grow_policy, nan_mode, sklearn_class, empty_trees
    )
    if scale != 1:
        bias = cb_model.get_scale_and_bias()[1]
        cb_model.set_scale_and_bias(scale, bias)
    if set_bias:
        scale = cb_model.get_scale_and_bias()[0]
        cb_model.set_scale_and_bias(scale, np.arange(3).astype(np.float64).tolist())
    d4p_model = d4p.mb.convert_model(cb_model)

    assert d4p_model.model_type == "catboost"
    assert d4p_model.is_classifier_
    assert d4p_model.n_classes_ == 3
    assert d4p_model.n_features_in_ == cb_model.n_features_in_

    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, cb_model.n_features_in_), dtype=dtype)
    if nan_mode != "Forbidden":
        X_test[:, 2:] = np.nan
        X_test[-1] = np.nan

    np.testing.assert_allclose(
        d4p_model.predict_proba(X_test),
        cb_model.predict(X_test, prediction_type="Probability"),
        atol=1e-5,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        d4p_model.predict(X_test),
        cb_model.predict(X_test, prediction_type="Class").reshape(-1),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.skipif(not cb_available, reason=cb_unavailable_str)
def test_catboost_default_objective():
    X, y = make_regression(n_samples=12, n_features=3, random_state=123)
    cb_model = cb.train(
        pool=cb.Pool(X, y),
        params={
            "depth": 2,
            "random_seed": 123,
            "thread_count": 1,
            "allow_writing_files": False,
        },
        num_boost_round=2,
        verbose=0,
        save_snapshot=False,
    )
    d4p_model = d4p.mb.convert_model(cb_model)
    np.testing.assert_allclose(
        d4p_model.predict(X),
        cb_model.predict(X, prediction_type="RawFormulaVal"),
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.skipif(not cb_available, reason=cb_unavailable_str)
def test_catboost_unsupported():
    X, y = make_regression(n_samples=10, n_features=2, n_targets=2, random_state=123)
    cb_model = cb.CatBoostRegressor(
        objective="MultiRMSE",
        depth=3,
        iterations=2,
        random_seed=123,
        thread_count=1,
        save_snapshot=False,
        verbose=0,
        allow_writing_files=False,
    ).fit(X, y)
    with pytest.raises(TypeError):
        d4p.mb.convert_model(cb_model)

    X, y = make_classification(
        n_samples=10,
        n_classes=2,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=123,
    )
    cb_model = cb.CatBoostClassifier(
        objective="MultiClassOneVsAll",
        depth=3,
        iterations=2,
        random_seed=123,
        thread_count=1,
        save_snapshot=False,
        verbose=0,
        allow_writing_files=False,
    ).fit(X, y)
    with pytest.raises(TypeError):
        d4p.mb.convert_model(cb_model)

    cb_model = cb.CatBoostClassifier(
        objective="MultiLogloss",
        depth=3,
        iterations=2,
        random_seed=123,
        thread_count=1,
        save_snapshot=False,
        verbose=0,
        allow_writing_files=False,
    ).fit(X, np.c_[y.reshape((-1, 1)), y[::-1].reshape((-1, 1))])
    with pytest.raises(TypeError):
        d4p.mb.convert_model(cb_model)

    X = X.astype(int)
    X -= X.min(axis=0, keepdims=True)
    cb_model = cb.CatBoostClassifier(
        objective="MultiClass",
        cat_features=np.arange(X.shape[1]).tolist(),
        depth=3,
        iterations=2,
        random_seed=123,
        thread_count=1,
        save_snapshot=False,
        verbose=0,
        allow_writing_files=False,
    ).fit(X, y)
    with pytest.raises(Exception):
        d4p.mb.convert_model(cb_model)


@pytest.mark.skip(reason="causes timeouts in CI")
def test_model_from_booster():
    class MockBooster:
        def get_dump(self, *_, **kwargs):
            # raw dump of 2 trees with a max depth of 1
            return [
                '  { "nodeid": 0, "depth": 0, "split": "1", "split_condition": 2, "yes": 1, "no": 2, "missing": 1 , "gain": 3, "cover": 4, "children": [\n    { "nodeid": 1, "leaf": 5 , "cover": 6 }, \n    { "nodeid": 2, "leaf": 7 , "cover":8 }\n  ]}',
                '  { "nodeid": 0, "leaf": 0.2 , "cover": 42 }',
            ]

    mock = MockBooster()
    result = gbt_convertors.TreeList.from_xgb_booster(
        mock, max_trees=0, feature_names_to_indices={"1": 1}
    )
    assert len(result) == 2

    tree0 = result[0]
    assert isinstance(tree0, gbt_convertors.TreeView)
    assert not tree0.is_leaf
    assert not hasattr(tree0, "cover")
    assert not hasattr(tree0, "value")

    assert isinstance(tree0.root_node, gbt_convertors.Node)

    assert tree0.root_node.cover == 4
    assert tree0.root_node.left_child.cover == 6
    assert tree0.root_node.right_child.cover == 8

    assert not tree0.root_node.is_leaf
    assert tree0.root_node.left_child.is_leaf
    assert tree0.root_node.right_child.is_leaf

    assert tree0.root_node.default_left
    assert not tree0.root_node.left_child.default_left
    assert not tree0.root_node.right_child.default_left

    assert tree0.root_node.feature == 1
    assert not hasattr(tree0.root_node.left_child, "feature")
    assert not hasattr(tree0.root_node.right_child, "feature")

    assert tree0.root_node.value == 2
    assert tree0.root_node.left_child.value == 5
    assert tree0.root_node.right_child.value == 7

    assert tree0.root_node.n_children == 2
    assert tree0.root_node.left_child.n_children == 0
    assert tree0.root_node.right_child.n_children == 0

    assert tree0.root_node.left_child.left_child is None
    assert tree0.root_node.left_child.right_child is None
    assert tree0.root_node.right_child.left_child is None
    assert tree0.root_node.right_child.right_child is None

    tree1 = result[1]
    assert isinstance(tree1, gbt_convertors.TreeView)
    assert tree1.is_leaf
    assert tree1.n_nodes == 1
    assert tree1.cover == 42
    assert tree1.value == 0.2


@pytest.mark.skip(reason="causes timeouts in CI")
@pytest.mark.parametrize("from_treelite", [False, True])
def test_unsupported_multiclass(from_treelite):
    X, y = make_classification(
        n_samples=10,
        n_classes=2,
        n_features=3,
        n_informative=3,
        n_redundant=0,
        random_state=123,
    )

    xgb_model = xgb.train(
        dtrain=xgb.DMatrix(X, y),
        num_boost_round=3,
        params={
            "objective": "multi:softprob",
            "num_class": 3,
            "multi_strategy": "multi_output_tree",
            "max_depth": 3,
            "seed": 123,
            "nthread": 1,
        },
    )
    # This currently fails on the XGB side due to being unable
    # to export to JSON. There's no explicit check for it in
    # the daal4py side. Once XGB implements the JSON dumping
    # functionality for multi-output trees, daal4py will need
    # to be modified to raise an error on such inputs.
    with pytest.raises(Exception):
        d4p.mb.convert_model(
            xgb_model if not from_treelite else treelite.frontend.from_xgboost(xgb_model)
        )

    lgb_model = lgb.train(
        train_set=lgb.Dataset(X, y),
        num_boost_round=3,
        params={
            "objective": "multiclassova",
            "num_class": 3,
            "num_leaves": 5,
            "num_threads": 1,
            "verbose": -1,
            "seed": 123,
            "deterministic": True,
        },
    )
    with pytest.raises(TypeError):
        d4p.mb.convert_model(
            lgb_model if not from_treelite else treelite.frontend.from_lightgbm(lgb_model)
        )


@pytest.mark.parametrize(
    "estimator,is_regression,expect_regressor,expect_warning,tl_predict",
    [
        (
            RandomForestRegressor(n_estimators=3),
            True,
            True,
            False,
            lambda tl_model, X: treelite.gtil.predict(
                tl_model, X, pred_margin=True
            ).reshape(-1),
        ),
        (
            GradientBoostingRegressor(n_estimators=3),
            True,
            True,
            False,
            lambda tl_model, X: treelite.gtil.predict(
                tl_model, X, pred_margin=True
            ).reshape(-1),
        ),
        (
            IsolationForest(n_estimators=3),
            True,
            True,
            False,
            lambda tl_model, X: treelite.gtil.predict(
                tl_model, X, pred_margin=True
            ).reshape(-1),
        ),
        (
            RandomForestClassifier(n_estimators=3),
            False,
            True,
            True,
            lambda tl_model, X: treelite.gtil.predict(tl_model, X, pred_margin=True)[
                :, 0, 1
            ],
        ),
        (
            GradientBoostingClassifier(n_estimators=3),
            False,
            False,
            False,
            lambda tl_model, X: treelite.gtil.predict(tl_model, X).reshape(-1),
        ),
    ],
)
def test_sklearn_through_treelite(
    estimator: BaseEstimator,
    is_regression: bool,
    expect_regressor: bool,
    expect_warning: bool,
    tl_predict: Callable,
):
    if is_regression:
        X, y = make_regression(n_samples=10, n_features=4, random_state=123)
    else:
        X, y = make_classification(
            n_samples=11,
            n_classes=2,
            n_features=3,
            n_informative=3,
            n_redundant=0,
            random_state=123,
        )
    X[:2] = (X[:2] * 10.0).astype(np.int32).astype(np.float64)
    tl_model = treelite.sklearn.import_model(estimator.fit(X, y))
    with pytest.warns() if expect_warning else contextlib.suppress():
        d4p_model = d4p.mb.convert_model(tl_model)

    if expect_regressor:
        assert d4p_model.is_regressor_
    else:
        assert d4p_model.is_classifier_

    tl_pred = tl_predict(tl_model, X)
    if d4p_model.is_regressor_:
        d4p_pred = d4p_model.predict(X)
    else:
        d4p_pred = d4p_model.predict_proba(X)[:, 1]

    np.testing.assert_allclose(
        d4p_pred,
        tl_pred,
        atol=1e-5,
        rtol=1e-5,
    )


def test_treelite_unsupported():
    if sklearn_check_version("1.4"):
        X, y = make_classification(
            n_samples=10,
            n_classes=3,
            n_informative=3,
            n_redundant=0,
            random_state=123,
        )
        tl_model = treelite.sklearn.import_model(
            RandomForestClassifier(n_estimators=3).fit(X, y)
        )
        with pytest.raises(TypeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d4p_model = d4p.mb.convert_model(tl_model)

        y_multi = np.c_[(y == 0).reshape((-1, 1)), (y == 1)[::-1].reshape((-1, 1))]
        tl_model = treelite.sklearn.import_model(
            RandomForestClassifier(n_estimators=3).fit(X, y_multi)
        )
        with pytest.raises(TypeError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d4p_model = d4p.mb.convert_model(tl_model)

    X, y = make_regression(n_samples=10, n_features=4, random_state=123, n_targets=2)
    tl_model = treelite.sklearn.import_model(
        RandomForestRegressor(n_estimators=3).fit(X, y)
    )
    with pytest.raises(TypeError):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d4p_model = d4p.mb.convert_model(tl_model)


# These aren't typically produced by the main libraries targeted by
# treelite, but can still be specified to be like this when constructing
# a model through their model builder.
@pytest.mark.parametrize("opname", [">", ">=", "<", "<="])
def test_treelite_uncommon(opname):
    # Taken from their example with a modified op:
    # https://treelite.readthedocs.io/en/latest/tutorials/builder.html
    builder = treelite.model_builder.ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=treelite.model_builder.Metadata(
            num_feature=3,
            task_type="kRegressor",
            average_tree_output=True,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=treelite.model_builder.TreeAnnotation(
            num_tree=1, target_id=[0], class_id=[0]
        ),
        postprocessor=treelite.model_builder.PostProcessorFunc(name="identity"),
        base_scores=[0.2],
    )
    builder.start_tree()
    builder.start_node(0)
    builder.numerical_test(
        feature_id=0,
        threshold=5.0,
        default_left=True,
        opname=opname,
        left_child_key=1,
        right_child_key=2,
    )
    builder.end_node()
    builder.start_node(1)
    builder.numerical_test(
        feature_id=2,
        threshold=-3.0,
        default_left=False,
        opname=opname,
        left_child_key=3,
        right_child_key=4,
    )
    builder.end_node()
    builder.start_node(2)
    builder.leaf(0.6)
    builder.end_node()
    builder.start_node(3)
    builder.leaf(-0.4)
    builder.end_node()
    builder.start_node(4)
    builder.leaf(1.2)
    builder.end_node()
    builder.end_tree()

    tl_model = builder.commit()
    d4p_model = d4p.mb.convert_model(tl_model)

    X = np.array(
        [
            [0.0, 0.0, -5.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, 1.0],
            [0.0, 5.0, -5.0],
            [0.0, 5.0, -2.0],
            [0.0, 5.0, 1.0],
            [10.0, 0.0, -5.0],
            [10.0, 0.0, -2.0],
            [10.0, 0.0, 1.0],
            [10.0, 5.0, -5.0],
            [10.0, 5.0, -2.0],
            [10.0, 5.0, 1.0],
        ]
    )
    np.testing.assert_almost_equal(
        d4p_model.predict(X),
        treelite.gtil.predict(tl_model, X).reshape(-1),
    )


def test_treelite_uneven_multiclass():
    # Also based on the same tutorial, with slight modifications:
    # https://treelite.readthedocs.io/en/latest/tutorials/builder.html
    builder = treelite.model_builder.ModelBuilder(
        threshold_type="float64",
        leaf_output_type="float64",
        metadata=treelite.model_builder.Metadata(
            num_feature=1,
            task_type="kMultiClf",
            average_tree_output=False,
            num_target=1,
            num_class=[3],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=treelite.model_builder.TreeAnnotation(
            num_tree=4,
            target_id=[0, 0, 0, 0],
            class_id=[0, 1, 2, 1],
        ),
        postprocessor=treelite.model_builder.PostProcessorFunc(name="softmax"),
        base_scores=[0.2, 0.0, 0.3],
    )
    for tree_id in range(4):
        builder.start_tree()
        builder.start_node(0)
        builder.numerical_test(
            feature_id=0,
            threshold=0.0,
            default_left=True,
            opname="<",
            left_child_key=1,
            right_child_key=2,
        )
        builder.end_node()
        builder.start_node(1)
        builder.leaf(0.5 if tree_id < 2 else 0.0)
        builder.end_node()
        builder.start_node(2)
        builder.leaf(1.0 if tree_id == 2 else 0.0)
        builder.end_node()
        builder.end_tree()
    tl_model = builder.commit()
    d4p_model = d4p.mb.convert_model(tl_model)

    X = np.array([[-1.0], [0.0], [1.0]])
    np.testing.assert_almost_equal(
        d4p_model.predict_proba(X),
        treelite.gtil.predict(tl_model, X)[:, 0, :],
    )


def test_sklearn_conversion_suggests_treelite():
    X, y = make_regression(n_samples=10, n_features=4, random_state=123)
    model = RandomForestRegressor(n_estimators=2).fit(X, y)
    with pytest.raises(TypeError, match="treelite"):
        d4p.mb.convert_model(model)


@pytest.mark.parametrize("with_names", [False, True])
@pytest.mark.parametrize("with_types", [False, True])
def test_xgb_object_is_not_corrupted(with_names, with_types):
    X, y = make_regression(n_samples=5, n_features=6, random_state=123)

    feature_types = None
    if with_types:
        X[:, 1] = X[:, 1].astype(int)
        feature_types = ["q", "int"] + (["q"] * (X.shape[1] - 2))

    feature_names = None
    if with_names:
        feature_names = [f"col{i+1}" for i in range(X.shape[1])]

    dm = xgb.DMatrix(
        X,
        y,
        feature_types=feature_types,
        feature_names=feature_names,
    )
    xgb_model = xgb.train(
        params={
            "objective": "reg:squarederror",
            "max_depth": 3,
            "seed": 123,
            "nthread": 1,
        },
        dtrain=dm,
        num_boost_round=3,
    )
    model_bytes_before = xgb_model.save_raw()

    d4p_model = d4p.mb.convert_model(xgb_model)

    model_bytes_after = xgb_model.save_raw()
    assert model_bytes_before == model_bytes_after

    xgb_pred = xgb_pred = xgb_model.predict(dm)
    xgb_pred_fresh = xgb.train(
        params={
            "objective": "reg:squarederror",
            "max_depth": 3,
            "seed": 123,
            "nthread": 1,
        },
        dtrain=xgb.DMatrix(X, y),
        num_boost_round=3,
    ).predict(xgb.DMatrix(X))
    np.testing.assert_almost_equal(xgb_pred, xgb_pred_fresh)

    np.testing.assert_allclose(d4p_model.predict(X), xgb_pred, rtol=1e-5)


# Note: there isn't any reason why these objects would get corrupted
# during the conversion, but there's no harm in testing just in case.
def test_lgb_model_is_not_corrupted():
    X, y = make_regression(n_samples=5, n_features=6, random_state=123)
    ds = lgb.Dataset(X, y)
    lgb_model = lgb.train(
        train_set=ds,
        num_boost_round=3,
        params={
            "objective": "regression",
            "num_leaves": 5,
            "num_threads": 1,
            "verbose": -1,
            "seed": 123,
            "deterministic": True,
        },
    )
    model_json_before = lgb_model.dump_model()

    d4p_model = d4p.mb.convert_model(lgb_model)

    model_json_after = lgb_model.dump_model()
    assert model_json_before == model_json_after

    lgb_pred = xgb_pred = lgb_model.predict(X)
    lgb_pred_fresh = lgb.train(
        train_set=lgb.Dataset(X, y),
        num_boost_round=3,
        params={
            "objective": "regression",
            "num_leaves": 5,
            "num_threads": 1,
            "verbose": -1,
            "seed": 123,
            "deterministic": True,
        },
    ).predict(X)
    np.testing.assert_almost_equal(lgb_pred, lgb_pred_fresh)

    np.testing.assert_allclose(d4p_model.predict(X), lgb_pred, rtol=1e-5)


def test_gbt_serialization():
    xgb_model = make_xgb_model("reg:gamma", None, False, False)
    d4p_model = d4p.mb.convert_model(xgb_model)
    rng = np.random.default_rng(seed=123)
    X_test = rng.standard_normal(size=(3, xgb_model.num_features()))
    pred_before = d4p_model.predict(X_test)

    d4p_bytes = pickle.dumps(d4p_model)
    del d4p_model, xgb_model
    gc.collect()

    d4p_new = pickle.loads(d4p_bytes)
    np.testing.assert_almost_equal(
        d4p_new.predict(X_test),
        pred_before,
    )


@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("n_classes", [2, 3])
def test_logreg_builder(fit_intercept, stochastic, n_classes):
    if stochastic:
        if not sklearn_check_version("1.1"):
            pytest.skip("Functionality introduced in a later sklearn version.")
        if n_classes != 2:
            pytest.skip("Functionality not yet implemented in sklearn.")
    if stochastic:
        model_skl = SGDClassifier(
            loss="log_loss", fit_intercept=fit_intercept, random_state=123
        )
    else:
        model_skl = LogisticRegression(fit_intercept=fit_intercept)

    X, y = make_classification(n_classes=n_classes, n_informative=4, random_state=123)
    model_skl.fit(X, y)

    model_d4p = d4p.mb.convert_model(model_skl)
    np.testing.assert_almost_equal(
        model_d4p.predict(X[::-1]),
        model_skl.predict(X[::-1]),
    )
    np.testing.assert_almost_equal(
        model_d4p.predict_proba(X[::-1]),
        model_skl.predict_proba(X[::-1]),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        np.testing.assert_almost_equal(
            model_d4p.predict_log_proba(X[::-1]),
            model_skl.predict_log_proba(X[::-1]),
        )

    np.testing.assert_almost_equal(
        model_d4p.coef_,
        model_skl.coef_,
    )
    np.testing.assert_almost_equal(
        model_d4p.intercept_,
        model_skl.intercept_,
    )


def test_logreg_builder_fp32():
    X, y = make_classification(random_state=123)
    model_skl = LogisticRegression().fit(X, y)
    model_d4p = d4p.mb.LogisticDAALModel(
        model_skl.coef_, model_skl.intercept_, dtype=np.float32
    )
    np.testing.assert_almost_equal(
        model_d4p.predict(X[::-1]),
        model_skl.predict(X[::-1]),
    )
    np.testing.assert_allclose(
        model_d4p.predict_proba(X[::-1]),
        model_skl.predict_proba(X[::-1]),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        model_d4p.predict_log_proba(X[::-1]),
        model_skl.predict_log_proba(X[::-1]),
        rtol=1e-3,
        atol=1e-6,
    )


def test_logreg_builder_serialization():
    X, y = make_classification(random_state=123)
    model_skl = LogisticRegression().fit(X, y)
    model_d4p_base = d4p.mb.convert_model(model_skl)
    model_d4p = pickle.loads(pickle.dumps(model_d4p_base))
    np.testing.assert_almost_equal(
        model_d4p_base.predict_proba(X[::-1]),
        model_d4p.predict_proba(X[::-1]),
    )


def test_logreg_builder_with_deleted_arrays():
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(5, 10))
    coefs = rng.standard_normal(size=(3, 10))
    intercepts = np.zeros(3)
    ref_pred = X @ coefs.T
    ref_probs = softmax(ref_pred, axis=1)

    model_d4p = d4p.mb.LogisticDAALModel(coefs, intercepts)
    coefs[:, :] = 0
    del coefs, intercepts
    gc.collect()

    np.testing.assert_almost_equal(
        model_d4p.predict_proba(X),
        ref_probs,
    )


# Note: these cases are safe to remove if scikit-learn later
# on decides to disallow some of these combinations.
@pytest.mark.parametrize(
    "estimator_skl,n_classes",
    [
        (
            LogisticRegression(multi_class="ovr"),
            3,
        ),
        (
            LogisticRegression(multi_class="multinomial"),
            2,
        ),
        # case below might change in the future if sklearn improves their modules
        pytest.param(
            SGDClassifier(loss="log_loss"),
            3,
            marks=pytest.mark.skipif(
                not sklearn_check_version("1.1"),
                reason="Requires higher sklearn version.",
            ),
        ),
        (
            SGDClassifier(loss="hinge"),
            2,
        ),
    ],
)
def test_logreg_builder_error_on_nonlogistic(estimator_skl, n_classes):
    X, y = make_classification(n_classes=n_classes, n_informative=4, random_state=123)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        estimator_skl.fit(X, y)

    with pytest.raises(TypeError):
        d4p.mb.convert_model(estimator_skl)

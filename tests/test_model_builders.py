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
import unittest
import warnings
from datetime import datetime

import lightgbm as lgbm
import numpy as np
import xgboost as xgb
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import daal4py as d4p
from daal4py.mb import gbt_convertors
from daal4py.sklearn._utils import daal_check_version

try:
    import catboost as cb

    cb_available = True
except ImportError:
    cb_available = False

try:
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
    catboost_skip_shap = tuple(map(int, cb.__version__.split("."))) < (1, 3, 0)
else:
    catboost_skip_shap = True
catboost_skip_shap_msg = (
    "CatBoost SHAP calculation is buggy. "
    "See https://github.com/catboost/catboost/issues/2556."
)


class LogRegModelBuilder(unittest.TestCase):
    def test_iris_with_intercept(self):
        X, y = load_iris(return_X_y=True)
        n_classes = 3
        clf = LogisticRegression(fit_intercept=True, max_iter=1000, random_state=0).fit(
            X, y
        )
        builder = d4p.logistic_regression_model_builder(
            n_classes=n_classes, n_features=X.shape[1]
        )
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    def test_iris_without_intercept(self):
        X, y = load_iris(return_X_y=True)
        n_classes = 3
        clf = LogisticRegression(fit_intercept=False, max_iter=1000, random_state=0).fit(
            X, y
        )
        builder = d4p.logistic_regression_model_builder(
            n_classes=n_classes, n_features=X.shape[1]
        )
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    def test_breast_cancer_with_intercept(self):
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = 2
        clf = LogisticRegression(fit_intercept=True, max_iter=10000, random_state=0).fit(
            X, y
        )
        builder = d4p.logistic_regression_model_builder(
            n_classes=n_classes, n_features=X.shape[1]
        )
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))

    def test_breast_cancer_without_intercept(self):
        X, y = load_breast_cancer(return_X_y=True)
        n_classes = 2
        clf = LogisticRegression(fit_intercept=False, max_iter=10000, random_state=0).fit(
            X, y
        )
        builder = d4p.logistic_regression_model_builder(
            n_classes=n_classes, n_features=X.shape[1]
        )
        builder.set_beta(clf.coef_, clf.intercept_)

        alg_pred = d4p.logistic_regression_prediction(nClasses=n_classes)

        pred_daal = alg_pred.compute(X, builder.model).prediction.flatten()
        pred_sklearn = clf.predict(X)
        self.assertTrue(np.allclose(pred_daal, pred_sklearn))


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostRegressionModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls, base_score=0.5):
        X, y = make_regression(n_samples=2, n_features=10, random_state=42)
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        cls.xgb_model = xgb.XGBRegressor(
            max_depth=5, n_estimators=50, random_state=42, base_score=base_score
        )
        cls.xgb_model.fit(X, y)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        self.assertEqual(m.model_type, "xgboost")
        # XGBoost treats regression as 0 classes, LightGBM 1 class
        # For us, it does not make a difference and both are acceptable
        self.assertEqual(m.n_classes_, 0)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_test)
        xgboost_pred = self.xgb_model.predict(self.X_test)
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-6)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_nan)
        xgboost_pred = self.xgb_model.predict(self.X_nan)
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-6)

    def test_model_predict_shap_contribs(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X_test, pred_contribs=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_test),
            pred_contribs=True,
            approx_contribs=False,
            validate_features=False,
        )
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-6)

    def test_model_predict_shap_interactions(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X_test, pred_interactions=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_test),
            pred_interactions=True,
            approx_contribs=False,
            validate_features=False,
        )
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-6)

    def test_model_predict_shap_contribs_missing_values(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X_nan, pred_contribs=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_nan),
            pred_contribs=True,
            approx_contribs=False,
            validate_features=False,
        )
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=5e-6)


# duplicate all tests for base_score=0.0
@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostRegressionModelBuilder_base_score0(XGBoostRegressionModelBuilder):
    @classmethod
    def setUpClass(cls):
        XGBoostRegressionModelBuilder.setUpClass(0)


# duplicate all tests for base_score=100
@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostRegressionModelBuilder_base_score100(XGBoostRegressionModelBuilder):
    @classmethod
    def setUpClass(cls):
        XGBoostRegressionModelBuilder.setUpClass(100)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostClassificationModelBuilder(unittest.TestCase):
    def get_conversion_warning_is_expected(self):
        if (
            self.xgb_model._estimator_type == "classifier"
            and self.xgb_model.objective == "binary:logitraw"
        ):
            return True
        else:
            return False

    @contextlib.contextmanager
    def conditional_assert_warns(self, warning):
        if warning is not None:
            with self.assertWarns(warning) as ctx:
                try:
                    yield [ctx]
                finally:
                    pass
        else:
            with warnings.catch_warnings() as ctx:
                warnings.simplefilter("error")
                try:
                    yield [ctx]
                finally:
                    pass

    @classmethod
    def setUpClass(cls, base_score=0.5, n_classes=2, objective="binary:logistic"):
        n_features = 15
        cls.base_score = base_score
        cls.n_classes = n_classes
        X, y = make_classification(
            n_samples=500,
            n_classes=n_classes,
            n_features=n_features,
            n_informative=(2 * n_features) // 3,
            random_state=42,
        )
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 2 * n_features, dtype=np.float32).reshape(
            2, n_features
        )
        cls.xgb_model = xgb.XGBClassifier(
            max_depth=5,
            n_estimators=50,
            random_state=42,
            base_score=base_score,
            objective=objective,
        )
        cls.xgb_model.fit(X, y)

    def test_model_conversion(self):
        expected_warning = (
            UserWarning if self.get_conversion_warning_is_expected() else None
        )
        with self.conditional_assert_warns(expected_warning):
            m = d4p.mb.convert_model(self.xgb_model.get_booster())
        self.assertEqual(m.model_type, "xgboost")
        self.assertEqual(m.n_classes_, self.n_classes)
        self.assertEqual(m.n_features_in_, 15)
        self.assertFalse(m._is_regression)

    def test_model_predict(self):
        expected_warning = (
            UserWarning if self.get_conversion_warning_is_expected() else None
        )
        with self.conditional_assert_warns(expected_warning):
            m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_test)
        xgboost_pred = self.xgb_model.predict(self.X_test)
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-7)

    def test_model_predict_proba(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict_proba(self.X_test)
        xgboost_pred = self.xgb_model.predict_proba(self.X_test)
        # calculating probas involves multiple exp / ln operations, therefore
        # they're quite susceptible to small numerical changes and we have to
        # accept an rtol of 1e-5
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-5)

    def test_missing_value_support(self):
        expected_warning = (
            UserWarning if self.get_conversion_warning_is_expected() else None
        )
        with self.conditional_assert_warns(expected_warning):
            m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_nan)
        xgboost_pred = self.xgb_model.predict(self.X_nan)
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-7)

    def test_model_predict_shap_contribs(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        if not shap_api_changed:
            with self.assertRaises(NotImplementedError):
                m.predict(self.X_test, pred_contribs=True)
        elif self.n_classes > 2:
            with self.assertRaisesRegex(
                RuntimeError, "Multiclass classification SHAP values not supported"
            ):
                m.predict(self.X_test, pred_contribs=True)
        else:
            d4p_pred = m.predict(self.X_test, pred_contribs=True)
            xgboost_pred = booster.predict(
                xgb.DMatrix(self.X_test),
                pred_contribs=True,
                approx_contribs=False,
                validate_features=False,
            )
            np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-5)

    def test_model_predict_shap_interactions(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        if not shap_api_changed:
            with self.assertRaises(NotImplementedError):
                m.predict(self.X_test, pred_contribs=True)
        elif self.n_classes > 2:
            with self.assertRaisesRegex(
                RuntimeError, "Multiclass classification SHAP values not supported"
            ):
                m.predict(self.X_test, pred_interactions=True)
        else:
            d4p_pred = m.predict(self.X_test, pred_interactions=True)
            xgboost_pred = booster.predict(
                xgb.DMatrix(self.X_test),
                pred_interactions=True,
                approx_contribs=False,
                validate_features=False,
            )
            # hitting floating precision limits for classification where class probabilities
            # are between 0 and 1
            # we need to accept large relative differences, as long as the absolute difference
            # remains small (<1e-6)
            np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=5e-2, atol=1e-6)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostGammaModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_regression(n_samples=2, n_features=10, random_state=42)
        y = np.exp((y - y.mean()) / y.std())
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        cls.xgb_model = xgb.XGBRegressor(
            max_depth=5, n_estimators=50, random_state=42, objective="reg:gamma"
        )
        cls.xgb_model.fit(X, y)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        self.assertEqual(m.model_type, "xgboost")
        self.assertEqual(m.n_classes_, 0)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_test)
        xgboost_pred = self.xgb_model.predict(self.X_test, output_margin=True)
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-6)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_nan)
        xgboost_pred = self.xgb_model.predict(self.X_nan, output_margin=True)
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-6)

    def test_model_predict_shap_contribs(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X_test, pred_contribs=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_test),
            pred_contribs=True,
            approx_contribs=False,
            validate_features=False,
        )
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-6)

    def test_model_predict_shap_interactions(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X_test, pred_interactions=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_test),
            pred_interactions=True,
            approx_contribs=False,
            validate_features=False,
        )
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-6)

    def test_model_predict_shap_contribs_missing_values(self):
        booster = self.xgb_model.get_booster()
        m = d4p.mb.convert_model(booster)
        d4p_pred = m.predict(self.X_nan, pred_contribs=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_nan),
            pred_contribs=True,
            approx_contribs=False,
            validate_features=False,
        )
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=5e-6)


# duplicate all tests for base_score=0.3
@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostClassificationModelBuilder_base_score03(XGBoostClassificationModelBuilder):
    @classmethod
    def setUpClass(cls):
        XGBoostClassificationModelBuilder.setUpClass(base_score=0.3)


# duplicate all tests for base_score=0.7
@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostClassificationModelBuilder_base_score07(XGBoostClassificationModelBuilder):
    @classmethod
    def setUpClass(cls):
        XGBoostClassificationModelBuilder.setUpClass(base_score=0.7)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostClassificationModelBuilder_n_classes5(XGBoostClassificationModelBuilder):
    @classmethod
    def setUpClass(cls):
        XGBoostClassificationModelBuilder.setUpClass(n_classes=5)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostClassificationModelBuilder_n_classes5_base_score03(
    XGBoostClassificationModelBuilder
):
    @classmethod
    def setUpClass(cls):
        XGBoostClassificationModelBuilder.setUpClass(n_classes=5, base_score=0.3)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostClassificationModelBuilder_objective_logitraw(
    XGBoostClassificationModelBuilder
):
    """
    Caveat: logitraw is not per se supported in daal4py because we always

                 1. normalize to probabilities ("activation") using sigmoid
                   (exception: SHAP values, the scores defining phi_ij are the raw class scores)

    However, by undoing the activation we can still compare if the original probas and SHAP values are aligned.
    """

    @classmethod
    def setUpClass(cls):
        XGBoostClassificationModelBuilder.setUpClass(
            n_classes=2, objective="binary:logitraw"
        )

    def test_model_predict_proba(self):
        with self.assertWarns(UserWarning):
            # expect a warning that logitraw behaves differently
            m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict_proba(self.X_test)

        # Note that XGBoost itself here will produce incorrect calculations for 'predict_proba'
        # due to the intercept being misapplied, but probabilities can correctly be obtained
        # by getting the raw prediction and transforming it to probabilities.
        xgboost_pred_raw = self.xgb_model.get_booster().inplace_predict(
            self.X_test, predict_type="margin"
        )
        xgboost_pred = 1 / (1 + np.exp(-xgboost_pred_raw))
        xgboost_pred = xgboost_pred.reshape((-1, 1))
        xgboost_pred = np.c_[1 - xgboost_pred, xgboost_pred]

        # calculating probas involves multiple exp / ln operations, therefore
        # they're quite susceptible to small numerical changes and we have to
        # accept an rtol of 1e-4
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=1e-4)

    @unittest.skipUnless(shap_api_changed, reason=shap_api_change_str)
    def test_model_predict_shap_contribs(self):
        booster = self.xgb_model.get_booster()
        with self.assertWarns(UserWarning):
            # expect a warning that logitraw behaves differently and/or
            # that base_score is ignored / fixed to 0.5
            m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_test, pred_contribs=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_test),
            pred_contribs=True,
            approx_contribs=False,
            validate_features=False,
        )
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=5e-6)

    @unittest.skipUnless(shap_api_changed, reason=shap_api_change_str)
    def test_model_predict_shap_interactions(self):
        booster = self.xgb_model.get_booster()
        with self.assertWarns(UserWarning):
            # expect a warning that logitraw behaves differently and/or
            # that base_score is ignored / fixed to 0.5
            m = d4p.mb.convert_model(self.xgb_model.get_booster())
        d4p_pred = m.predict(self.X_test, pred_interactions=True)
        xgboost_pred = booster.predict(
            xgb.DMatrix(self.X_test),
            pred_interactions=True,
            approx_contribs=False,
            validate_features=False,
        )
        np.testing.assert_allclose(d4p_pred, xgboost_pred, rtol=5e-5)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class LightGBMRegressionModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        # LightGBM requires a couple of NaN values in the training data to properly set
        # the missing value type to NaN
        # https://github.com/microsoft/LightGBM/issues/6139
        X_train = np.concatenate([cls.X_nan, X])
        y_train = np.concatenate([[0, 0], y])
        params = {
            "task": "train",
            "boosting": "gbdt",
            "objective": "regression",
            "num_leaves": 4,
            "learning_rage": 0.05,
            "metric": {"l2", "l1"},
            "verbose": -1,
        }
        cls.lgbm_model = lgbm.train(
            params,
            train_set=lgbm.Dataset(X_train, y_train),
            num_boost_round=1,
        )

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        self.assertEqual(m.model_type, "lightgbm")
        self.assertEqual(m.n_classes_, 1)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_test)
        lgbm_pred = self.lgbm_model.predict(self.X_test)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-6)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_nan)
        lgbm_pred = self.lgbm_model.predict(self.X_nan)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=5e-6)

    def test_model_predict_shap_contribs(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_test, pred_contribs=True)
        lgbm_pred = self.lgbm_model.predict(self.X_test, pred_contrib=True)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-6)

    @unittest.skipUnless(shap_available, reason=shap_unavailable_str)
    def test_model_predict_shap_interactions(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        # SHAP Python package drops bias terms from the returned matrix, therefore we drop the final row & column
        d4p_pred = m.predict(self.X_test, pred_interactions=True)[:, :-1, :-1]
        explainer = shap.TreeExplainer(self.lgbm_model)
        shap_pred = explainer.shap_interaction_values(self.X_test)
        np.testing.assert_allclose(d4p_pred, shap_pred, rtol=1e-6)

    def test_model_predict_shap_contribs_missing_values(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_nan, pred_contribs=True)
        lgbm_pred = self.lgbm_model.predict(self.X_nan, pred_contrib=True)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-6)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class LightGBMClassificationModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            random_state=3, n_classes=3, n_informative=3, n_features=10
        )
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        X_train = np.concatenate([cls.X_nan, X])
        y_train = np.concatenate([[0, 0], y])
        params = {
            "task": "train",
            "boosting": "gbdt",
            "objective": "multiclass",
            "num_leaves": 4,
            "num_class": 3,
            "verbose": -1,
        }
        cls.lgbm_model = lgbm.train(
            params,
            train_set=lgbm.Dataset(X_train, y_train),
            num_boost_round=10,
        )

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        self.assertEqual(m.model_type, "lightgbm")
        self.assertEqual(m.n_classes_, 3)
        self.assertEqual(m.n_features_in_, 10)
        self.assertFalse(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_test)
        lgbm_pred = np.argmax(self.lgbm_model.predict(self.X_test), axis=1)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-7)

    def test_model_predict_proba(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict_proba(self.X_test)
        lgbm_pred = self.lgbm_model.predict(self.X_test)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-7)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_nan)
        lgbm_pred = np.argmax(self.lgbm_model.predict(self.X_nan), axis=1)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-7)

    def test_model_predict_shap_contribs(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_test, pred_contribs=True)

    def test_model_predict_shap_interactions(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_test, pred_interactions=True)

    def test_model_predict_shap_contribs_missing_values(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_nan, pred_contribs=True)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class LightGBMClassificationModelBuilder_binaryClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            random_state=3, n_classes=2, n_informative=3, n_features=10
        )
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        X_train = np.concatenate([cls.X_nan, X])
        y_train = np.concatenate([[0, 0], y])
        params = {
            "task": "train",
            "boosting": "gbdt",
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 4,
            "verbose": -1,
        }
        cls.lgbm_model = lgbm.train(
            params,
            train_set=lgbm.Dataset(X_train, y_train),
            num_boost_round=10,
        )

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        self.assertEqual(m.model_type, "lightgbm")
        self.assertEqual(m.n_classes_, 2)
        self.assertEqual(m.n_features_in_, 10)
        self.assertFalse(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_test)
        lgbm_pred = np.round(self.lgbm_model.predict(self.X_test)).astype(int)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-7)

    def test_model_predict_proba(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        # predict proba of being class 1
        d4p_pred = m.predict_proba(self.X_test)[:, 1]
        lgbm_pred = self.lgbm_model.predict(self.X_test)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-7)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        d4p_pred = m.predict(self.X_nan)
        lgbm_pred = np.round(self.lgbm_model.predict(self.X_nan)).astype(int)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-7)

    def test_model_predict_proba_missing_values(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        # predict proba of being class 1
        d4p_pred = m.predict_proba(self.X_nan)[:, 1]
        lgbm_pred = self.lgbm_model.predict(self.X_nan)
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-7)

    def test_model_predict_shap_contribs(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_test, pred_contribs=True)

    def test_model_predict_shap_interactions(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_test, pred_interactions=True)

    def test_model_predict_shap_contribs_missing_values(self):
        m = d4p.mb.convert_model(self.lgbm_model)
        with self.assertRaises(NotImplementedError):
            m.predict(self.X_nan, pred_contribs=True)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
@unittest.skipUnless(cb_available, reason=cb_unavailable_str)
class CatBoostRegressionModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        cls.X_test = X[:2, :]
        cls.y_test = y[:2]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        params = {
            "reg_lambda": 1,
            "max_depth": 3,
            "num_leaves": 2**3,
            "verbose": 0,
            "objective": "RMSE",
            "learning_rate": 0.3,
            "n_estimators": 25,
        }
        cls.cb_model = cb.CatBoost(params)
        cls.cb_model.fit(X, y, verbose=0)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.cb_model)
        self.assertTrue(hasattr(m, "daal_model_"))
        self.assertIsInstance(m.daal_model_, d4p._daal4py.gbt_regression_model)
        self.assertEqual(m.model_type, "catboost")
        self.assertEqual(m.daal_model_.NumberOfFeatures, 10)
        self.assertEqual(m.daal_model_.NumberOfTrees, 25)
        self.assertEqual(m.n_features_in_, 10)
        self.assertTrue(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_test)
        cb_pred = self.cb_model.predict(self.X_test)
        np.testing.assert_allclose(d4p_pred, cb_pred, rtol=1e-7)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_nan)
        cb_pred = self.cb_model.predict(self.X_nan)
        np.testing.assert_allclose(d4p_pred, cb_pred, rtol=1e-7)

    @unittest.skipIf(catboost_skip_shap, reason=catboost_skip_shap_msg)
    def test_model_predict_shap_contribs(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_test, pred_contribs=True)
        lgbm_pred = self.cb_model.get_feature_importance(
            cb.Pool(self.X_test, self.y_test), type="ShapValues"
        )
        np.testing.assert_allclose(d4p_pred, lgbm_pred, rtol=1e-6)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
@unittest.skipUnless(cb_available, reason=cb_unavailable_str)
class CatBoostClassificationModelBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        X, y = make_classification(
            n_classes=3, n_features=10, n_informative=3, random_state=42
        )
        cls.X_test = X[:2, :]
        cls.X_nan = np.array([np.nan] * 20, dtype=np.float32).reshape(2, 10)
        params = {
            "reg_lambda": 1,
            "max_depth": 3,
            "num_leaves": 2**3,
            "verbose": 0,
            "objective": "MultiClass",
            "learning_rate": 0.3,
            "n_estimators": 25,
        }
        cls.cb_model = cb.CatBoost(params)
        cls.cb_model.fit(X, y, verbose=0)

    def test_model_conversion(self):
        m = d4p.mb.convert_model(self.cb_model)
        self.assertTrue(hasattr(m, "daal_model_"))
        self.assertIsInstance(m.daal_model_, d4p._daal4py.gbt_classification_model)
        self.assertEqual(m.model_type, "catboost")
        self.assertEqual(m.daal_model_.NumberOfFeatures, 10)
        self.assertEqual(m.daal_model_.NumberOfTrees, 3 * 25)
        self.assertEqual(m.n_features_in_, 10)
        self.assertFalse(m._is_regression)

    def test_model_predict(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_test)
        cb_pred = self.cb_model.predict(self.X_test, prediction_type="Class").T[0]
        np.testing.assert_allclose(d4p_pred, cb_pred, rtol=1e-7)

    def test_missing_value_support(self):
        m = d4p.mb.convert_model(self.cb_model)
        d4p_pred = m.predict(self.X_nan)
        cb_pred = self.cb_model.predict(self.X_nan, prediction_type="Class").T[0]
        np.testing.assert_allclose(d4p_pred, cb_pred, rtol=1e-7)

    def test_model_predict_shap_contribs(self):
        # SHAP value support from CatBoost models is to be added
        with self.assertWarnsRegex(
            Warning,
            "Converted models of this type do not support SHAP value calculation",
        ):
            d4p.mb.convert_model(self.cb_model)


@unittest.skipUnless(shap_supported, reason=shap_not_supported_str)
class XGBoostEarlyStopping(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        num_classes = 3
        X, y = make_classification(
            n_samples=1500,
            n_features=10,
            n_informative=3,
            n_classes=num_classes,
            random_state=42,
        )
        X_train, cls.X_test, y_train, cls.y_test = train_test_split(
            X, y, test_size=0.5, random_state=42
        )

        # training parameters setting
        params = {
            "n_estimators": 100,
            "max_bin": 256,
            "scale_pos_weight": 2,
            "lambda_l2": 1,
            "alpha": 0.9,
            "max_depth": 8,
            "num_leaves": 2**8,
            "verbosity": 0,
            "objective": "multi:softproba",
            "learning_rate": 0.3,
            "num_class": num_classes,
            "early_stopping_rounds": 5,
            "verbose_eval": False,
        }

        cls.xgb_clf = xgb.XGBClassifier(**params)
        cls.xgb_clf.fit(
            X_train, y_train, eval_set=[(cls.X_test, cls.y_test)], verbose=False
        )
        cls.daal_model = d4p.mb.convert_model(cls.xgb_clf.get_booster())

    def test_early_stopping(self):
        xgb_prediction = self.xgb_clf.predict(self.X_test)
        xgb_proba = self.xgb_clf.predict_proba(self.X_test)
        xgb_errors_count = np.count_nonzero(xgb_prediction - np.ravel(self.y_test))

        daal_prediction = self.daal_model.predict(self.X_test)
        daal_proba = self.daal_model.predict_proba(self.X_test)
        daal_errors_count = np.count_nonzero(daal_prediction - np.ravel(self.y_test))

        self.assertTrue(np.absolute(xgb_errors_count - daal_errors_count) == 0)

        np.testing.assert_allclose(xgb_proba, daal_proba, rtol=1e-6)


class ModelBuilderTreeView(unittest.TestCase):
    def test_model_from_booster(self):
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
        self.assertEqual(len(result), 2)

        tree0 = result[0]
        self.assertIsInstance(tree0, gbt_convertors.TreeView)
        self.assertFalse(tree0.is_leaf)
        with self.assertRaises(ValueError):
            tree0.cover
        with self.assertRaises(ValueError):
            tree0.value

        self.assertIsInstance(tree0.root_node, gbt_convertors.Node)

        self.assertEqual(tree0.root_node.cover, 4)
        self.assertEqual(tree0.root_node.left_child.cover, 6)
        self.assertEqual(tree0.root_node.right_child.cover, 8)

        self.assertFalse(tree0.root_node.is_leaf)
        self.assertTrue(tree0.root_node.left_child.is_leaf)
        self.assertTrue(tree0.root_node.right_child.is_leaf)

        self.assertTrue(tree0.root_node.default_left)
        self.assertFalse(tree0.root_node.left_child.default_left)
        self.assertFalse(tree0.root_node.right_child.default_left)

        self.assertEqual(tree0.root_node.feature, 1)
        with self.assertRaises(ValueError):
            tree0.root_node.left_child.feature
        with self.assertRaises(ValueError):
            tree0.root_node.right_child.feature

        self.assertEqual(tree0.root_node.value, 2)
        self.assertEqual(tree0.root_node.left_child.value, 5)
        self.assertEqual(tree0.root_node.right_child.value, 7)

        self.assertEqual(tree0.root_node.n_children, 2)
        self.assertEqual(tree0.root_node.left_child.n_children, 0)
        self.assertEqual(tree0.root_node.right_child.n_children, 0)

        self.assertIsNone(tree0.root_node.left_child.left_child)
        self.assertIsNone(tree0.root_node.left_child.right_child)
        self.assertIsNone(tree0.root_node.right_child.left_child)
        self.assertIsNone(tree0.root_node.right_child.right_child)

        tree1 = result[1]
        self.assertIsInstance(tree1, gbt_convertors.TreeView)
        self.assertTrue(tree1.is_leaf)
        self.assertEqual(tree1.n_nodes, 1)
        self.assertEqual(tree1.cover, 42)
        self.assertEqual(tree1.value, 0.2)


class TestXGBObjectIsNotCorrupted(unittest.TestCase):
    def test_xgb_not_corrupted_no_names(self):
        X, y = make_regression(n_samples=100, n_features=10, random_state=123)
        X[:, 1] = X[:, 1].astype(int)
        dm = xgb.DMatrix(X, y, feature_types=["q", "int"] + (["q"] * (X.shape[1] - 2)))
        xgb_model = xgb.train(
            params={"objective": "reg:squarederror", "seed": 123},
            dtrain=dm,
            num_boost_round=10,
        )
        model_bytes_before = xgb_model.save_raw()

        d4p_model = d4p.mb.convert_model(xgb_model)

        model_bytes_after = xgb_model.save_raw()
        assert model_bytes_before == model_bytes_after

        xgb_pred = xgb_pred = xgb_model.predict(dm)
        xgb_pred_fresh = xgb.train(
            params={"objective": "reg:squarederror", "seed": 123},
            dtrain=xgb.DMatrix(X, y),
            num_boost_round=10,
        ).predict(xgb.DMatrix(X))
        np.testing.assert_almost_equal(xgb_pred, xgb_pred_fresh)

    def test_xgb_not_corrupted_with_names(self):
        X, y = make_regression(n_samples=100, n_features=10, random_state=123)
        X[:, 1] = X[:, 1].astype(int)
        dm = xgb.DMatrix(
            X,
            y,
            feature_types=["q", "int"] + (["q"] * (X.shape[1] - 2)),
            feature_names=[f"col{i+1}" for i in range(X.shape[1])],
        )
        xgb_model = xgb.train(
            params={"objective": "reg:squarederror", "seed": 123},
            dtrain=dm,
            num_boost_round=10,
        )
        model_bytes_before = xgb_model.save_raw()

        d4p_model = d4p.mb.convert_model(xgb_model)

        model_bytes_after = xgb_model.save_raw()
        assert model_bytes_before == model_bytes_after

        xgb_pred = xgb_pred = xgb_model.predict(dm)
        xgb_pred_fresh = xgb.train(
            params={"objective": "reg:squarederror", "seed": 123},
            dtrain=xgb.DMatrix(X, y),
            num_boost_round=10,
        ).predict(xgb.DMatrix(X))
        np.testing.assert_almost_equal(xgb_pred, xgb_pred_fresh)

        np.testing.assert_allclose(d4p_model.predict(X), xgb_pred, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()

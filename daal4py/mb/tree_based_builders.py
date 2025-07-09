# ==============================================================================
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
# ==============================================================================

# daal4py Model builders API

import warnings
from typing import Literal, Optional

import numpy as np

import daal4py as d4p

try:
    from pandas import DataFrame
    from pandas.core.dtypes.cast import find_common_type

    pandas_is_imported = True
except (ImportError, ModuleNotFoundError):
    pandas_is_imported = False

from sklearn.utils.metaestimators import available_if

from .gbt_convertors import (
    get_catboost_params,
    get_gbt_model_from_catboost,
    get_gbt_model_from_lightgbm,
    get_gbt_model_from_treelite,
    get_gbt_model_from_xgboost,
    get_lightgbm_params,
    get_xgboost_params,
)


def parse_dtype(dt):
    if dt == np.double:
        return "double"
    if dt == np.single:
        return "float"
    raise ValueError(f"Input array has unexpected dtype = {dt}")


def getFPType(X):
    if pandas_is_imported:
        if isinstance(X, DataFrame):
            dt = find_common_type(X.dtypes.tolist())
            return parse_dtype(dt)

    dt = getattr(X, "dtype", None)
    return parse_dtype(dt)


class GBTDAALBaseModel:
    def __init__(self):
        self.model_type: Optional[
            Literal["xgboost", "catboost", "lightgbm", "treelite"]
        ] = None

    @property
    def _is_regression(self):
        return hasattr(self, "daal_model_") and isinstance(
            self.daal_model_, d4p.gbt_regression_model
        )

    def _get_params_from_lightgbm(self, params):
        self.n_classes_ = params["num_tree_per_iteration"]
        objective_fun = params["objective"]
        if self.n_classes_ <= 2:
            if "binary" in objective_fun:  # nClasses == 1
                self.n_classes_ = 2

        self.n_features_in_ = params["max_feature_idx"] + 1

    def _get_params_from_xgboost(self, params):
        self.n_classes_ = int(params["learner"]["learner_model_param"]["num_class"])
        objective_fun = params["learner"]["learner_train_param"]["objective"]
        if self.n_classes_ <= 2:
            if objective_fun in ["binary:logistic", "binary:logitraw"]:
                self.n_classes_ = 2
            elif self.n_classes_ == 0:
                self.n_classes_ = 1

        self.n_features_in_ = int(params["learner"]["learner_model_param"]["num_feature"])

    def _get_params_from_catboost(self, params):
        if "class_params" in params["model_info"]:
            self.n_classes_ = len(params["model_info"]["class_params"]["class_to_label"])
        else:
            self.n_classes_ = 1
        self.n_features_in_ = len(params["features_info"]["float_features"])

    def _convert_model_from_lightgbm(self, booster):
        lgbm_params = get_lightgbm_params(booster)
        self.daal_model_ = get_gbt_model_from_lightgbm(booster, lgbm_params)
        self._get_params_from_lightgbm(lgbm_params)
        self.supports_shap_ = self.n_classes_ < 3

    def _convert_model_from_xgboost(self, booster):
        xgb_params = get_xgboost_params(booster)
        self.daal_model_ = get_gbt_model_from_xgboost(booster, xgb_params)
        self._get_params_from_xgboost(xgb_params)
        self.supports_shap_ = self.n_classes_ < 3

    def _convert_model_from_catboost(self, booster):
        catboost_params = get_catboost_params(booster)
        self.daal_model_, self.supports_shap_ = get_gbt_model_from_catboost(booster)
        self._get_params_from_catboost(catboost_params)

    def _convert_model_from_treelite(self, tl_model):
        self.daal_model_, self.n_classes_, self.n_features_in_, self.supports_shap_ = (
            get_gbt_model_from_treelite(tl_model)
        )

    def _convert_model(self, model):
        (submodule_name, class_name) = (
            model.__class__.__module__,
            model.__class__.__name__,
        )
        self_class_name = self.__class__.__name__

        # Build GBTDAALClassifier from LightGBM
        if (submodule_name, class_name) == ("lightgbm.sklearn", "LGBMClassifier"):
            self._convert_model_from_lightgbm(model.booster_)
        # Build GBTDAALClassifier from XGBoost
        elif (submodule_name, class_name) == ("xgboost.sklearn", "XGBClassifier"):
            self._convert_model_from_xgboost(model.get_booster())
        # Build GBTDAALClassifier from CatBoost
        elif (submodule_name, class_name) == ("catboost.core", "CatBoostClassifier"):
            self._convert_model_from_catboost(model)
        # Build GBTDAALRegressor from LightGBM
        elif (submodule_name, class_name) == ("lightgbm.sklearn", "LGBMRegressor"):
            self._convert_model_from_lightgbm(model.booster_)
        # Build GBTDAALRegressor from XGBoost
        elif (submodule_name, class_name) == ("xgboost.sklearn", "XGBRegressor"):
            self._convert_model_from_xgboost(model.get_booster())
        # Build GBTDAALRegressor from CatBoost
        elif (submodule_name, class_name) == ("catboost.core", "CatBoostRegressor"):
            self._convert_model_from_catboost(model)
        # Build GBTDAALModel from LightGBM
        elif (submodule_name, class_name) == ("lightgbm.basic", "Booster"):
            self._convert_model_from_lightgbm(model)
        # Build GBTDAALModel from XGBoost
        elif (submodule_name, class_name) == ("xgboost.core", "Booster"):
            self._convert_model_from_xgboost(model)
        # Build GBTDAALModel from CatBoost
        elif (submodule_name, class_name) == ("catboost.core", "CatBoost"):
            self._convert_model_from_catboost(model)
        elif (submodule_name, class_name) == ("treelite.model", "Model"):
            self._convert_model_from_treelite(model)
        elif submodule_name.startswith("sklearn.ensemble"):
            raise TypeError(
                "Cannot convert scikit-learn models. Try converting to treelite "
                "with 'treelite.sklearn.import_model' and then converting the "
                "resulting TreeLite object."
            )
        else:
            raise TypeError(f"Unknown model format {submodule_name}.{class_name}")

    def _predict_classification(
        self, X, fptype, resultsToEvaluate, pred_contribs=False, pred_interactions=False
    ):
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Shape of input is different from what was seen in `fit`")

        if not hasattr(self, "daal_model_"):
            raise ValueError(
                (
                    "The class {} instance does not have 'daal_model_' attribute set. "
                    "Call 'fit' with appropriate arguments before using this method."
                ).format(type(self).__name__)
            )

        # Prediction
        try:
            return self._predict_classification_with_results_to_compute(
                X, fptype, resultsToEvaluate, pred_contribs, pred_interactions
            )
        except TypeError as e:
            if "unexpected keyword argument 'resultsToCompute'" in str(e):
                if pred_contribs or pred_interactions:
                    # SHAP values requested, but not supported by this version
                    raise TypeError(
                        f"{'pred_contribs' if pred_contribs else 'pred_interactions'} not supported by this version of daal4py"
                    ) from e
            else:
                # unknown type error
                raise
        except RuntimeError as e:
            if "Method is not implemented" in str(e):
                if pred_contribs or pred_interactions:
                    raise NotImplementedError(
                        f"{'pred_contribs' if pred_contribs else 'pred_interactions'} is not implemented for classification models"
                    )
            else:
                raise

        # fallback to calculation without `resultsToCompute`
        predict_algo = d4p.gbt_classification_prediction(
            nClasses=self.n_classes_,
            fptype=fptype,
            resultsToEvaluate=resultsToEvaluate,
        )
        predict_result = predict_algo.compute(X, self.daal_model_)

        if resultsToEvaluate == "computeClassLabels":
            return predict_result.prediction.ravel().astype(np.int64, copy=False)
        else:
            return predict_result.probabilities

    def _predict_classification_with_results_to_compute(
        self,
        X,
        fptype,
        resultsToEvaluate,
        pred_contribs=False,
        pred_interactions=False,
    ):
        """Assume daal4py supports the resultsToCompute kwarg"""
        resultsToCompute = ""
        if pred_contribs:
            resultsToCompute = "shapContributions"
        elif pred_interactions:
            resultsToCompute = "shapInteractions"

        predict_algo = d4p.gbt_classification_prediction(
            nClasses=self.n_classes_,
            fptype=fptype,
            resultsToCompute=resultsToCompute,
            resultsToEvaluate=resultsToEvaluate,
        )
        predict_result = predict_algo.compute(X, self.daal_model_)

        if pred_contribs:
            return predict_result.prediction.ravel().reshape((-1, X.shape[1] + 1))
        elif pred_interactions:
            return predict_result.prediction.ravel().reshape(
                (-1, X.shape[1] + 1, X.shape[1] + 1)
            )
        elif resultsToEvaluate == "computeClassLabels":
            return predict_result.prediction.ravel().astype(np.int64, copy=False)
        else:
            return predict_result.probabilities

    def _predict_regression(
        self, X, fptype, pred_contribs=False, pred_interactions=False
    ):
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Shape of input is different from what was seen in `fit`")

        if not hasattr(self, "daal_model_"):
            raise ValueError(
                (
                    "The class {} instance does not have 'daal_model_' attribute set. "
                    "Call 'fit' with appropriate arguments before using this method."
                ).format(type(self).__name__)
            )

        try:
            return self._predict_regression_with_results_to_compute(
                X, fptype, pred_contribs, pred_interactions
            )
        except TypeError as e:
            if "unexpected keyword argument 'resultsToCompute'" in str(e) and (
                pred_contribs or pred_interactions
            ):
                # SHAP values requested, but not supported by this version
                raise TypeError(
                    f"{'pred_contribs' if pred_contribs else 'pred_interactions'} not supported by this version of daalp4y"
                ) from e
            else:
                # unknown type error
                raise

    def _predict_regression_with_results_to_compute(
        self, X, fptype, pred_contribs=False, pred_interactions=False
    ):
        """Assume daal4py supports the resultsToCompute kwarg"""
        resultsToCompute = ""
        if pred_contribs:
            resultsToCompute = "shapContributions"
        elif pred_interactions:
            resultsToCompute = "shapInteractions"

        predict_algo = d4p.gbt_regression_prediction(
            fptype=fptype, resultsToCompute=resultsToCompute
        )
        predict_result = predict_algo.compute(X, self.daal_model_)

        if pred_contribs:
            return predict_result.prediction.ravel().reshape((-1, X.shape[1] + 1))
        elif pred_interactions:
            return predict_result.prediction.ravel().reshape(
                (-1, X.shape[1] + 1, X.shape[1] + 1)
            )
        else:
            return predict_result.prediction.ravel()


class GBTDAALModel(GBTDAALBaseModel):
    """
    Gradient Boosted Decision Tree Model

    Model class offering accelerated predictions for gradient-boosted decision
    tree models from other libraries.

    Objects of this class are meant to be initialized from GBT model objects
    created through other libraries, returning a different class which can calculate
    predictions faster than the original library that created said model.

    Can be created from model objects that meet all of the following criteria:

    - Were produced from one of the following libraries: ``xgboost``, ``lightgbm``, ``catboost``,
      or ``treelite`` (with some limitations). It can work with either the base booster classes
      of those libraries or with their scikit-learn-compatible classes.
    - Do not use categorical features.
    - Are for regression or classification (e.g. no ranking). In the case of XGBoost objective
      ``binary:logitraw``, it will create a classification model out of it, and in the case of
      objective ``reg:logistic``, will create a regression model.
    - Are not multi-output models. Note that multi-class classification **is** supported.
    - Are not multi-class random forests (multi-class gradient boosters are supported).

    Note that while models from packages such as scikit-learn are not supported directly,
    they can still be converted to this class by first converting them to TreeLite and
    then converting to :obj:`GBTDAALModel` from that TreeLite model. In such case, note that
    models corresponding to random forest binary classifiers will be treated as regressors
    that predict probabilities.

    Parameters
    ----------
    model : booster object from another library
        The fitted GBT model from which this object will be created. See rest of the documentation
        for supported input types.

    Attributes
    ----------
    is_classifier_ : bool
        Whether this is a classification model.
    is_regressor_ : bool
        Whether this is a regression model.
    supports_shap_ : bool
        Whether the model supports SHAP calculations.
    """

    def __init__(self, model):
        self._convert_model(model)
        for type_str in ("xgboost", "lightgbm", "catboost", "treelite"):
            if type_str in str(type(model)):
                self.model_type = type_str
                break

    def predict(
        self, X, pred_contribs: bool = False, pred_interactions: bool = False
    ) -> np.ndarray:
        """
        Compute model predictions on new data

        Computes the predicted values of the response variable for new data given the features / covariates
        for each row.

        In the case of classification models, this will output the most probable class (see
        :meth:`predict_proba` for probability predictions), while in the case of regression
        models, will output values in the link scale (what XGBoost calls 'margin' and LightGBM
        calls 'raw').

        :param X: The features covariates. Should be an array of shape ``[num_samples, num_features]``.
        :param bool pred_contribs: Whether to predict feature contributions. Result should have shape ``[num_samples, num_features+1]``, with the last column corresponding to the intercept. See :obj:`xgboost.Booster.predict` for more details about this type of computation.
        :param bool pred_interactions: Whether to predict feature interactions. Result should have shape ``[num_samples, num_features+1, num_features+1]``, with the last position across the last two dimensions corresponding to the intercept. See :obj:`xgboost.Booster.predict` for more details about this type of computation.

        :rtype: np.ndarray
        """
        if pred_contribs or pred_interactions:
            if not self.supports_shap_:
                raise TypeError("SHAP calculations are not available for this model.")
            if self.model_type == "catboost":
                warnings.warn(
                    "SHAP values from models converted from CatBoost do not match "
                    "against those of the original library. See "
                    "https://github.com/catboost/catboost/issues/2556 for more details."
                )
        fptype = getFPType(X)
        if self._is_regression:
            return self._predict_regression(X, fptype, pred_contribs, pred_interactions)
        else:
            return self._predict_classification(
                X, fptype, "computeClassLabels", pred_contribs, pred_interactions
            )

    @property
    def is_classifier_(self) -> bool:
        """Whether this is a classification model"""
        return not self._is_regression

    @property
    def is_regressor_(self) -> bool:
        """Whether this is a regression model"""
        return self._is_regression

    def _check_proba(self):
        return not self._is_regression

    @available_if(_check_proba)
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities

        Computes the predicted probabilities of belonging to each class for each row in the
        input data given the features / covariates. Output shape is ``[num_samples, num_classes]``.

        :param X: The features covariates. Should be an array of shape ``[num_samples, num_features]``.
        :rtype: np.ndarray
        """
        fptype = getFPType(X)
        return self._predict_classification(X, fptype, "computeClassProbabilities")

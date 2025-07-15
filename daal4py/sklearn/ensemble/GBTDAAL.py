# ==============================================================================
# Copyright 2014 Intel Corporation
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

# daal4py GBT scikit-learn-compatible estimator class

import numbers

import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import daal4py as d4p
from daal4py.sklearn._utils import sklearn_check_version

from .._n_jobs_support import control_n_jobs
from .._utils import getFPType
from ..utils.validation import validate_data


class GBTDAALBase(BaseEstimator, d4p.mb.GBTDAALBaseModel):
    def __init__(
        self,
        split_method="inexact",
        max_iterations=50,
        max_tree_depth=6,
        shrinkage=0.3,
        min_split_loss=0,
        reg_lambda=1,
        observations_per_tree_fraction=1,
        features_per_node=0,
        min_observations_in_leaf_node=5,
        memory_saving_mode=False,
        max_bins=256,
        min_bin_size=5,
        random_state=None,
    ):
        self.split_method = split_method
        self.max_iterations = max_iterations
        self.max_tree_depth = max_tree_depth
        self.shrinkage = shrinkage
        self.min_split_loss = min_split_loss
        self.reg_lambda = reg_lambda
        self.observations_per_tree_fraction = observations_per_tree_fraction
        self.features_per_node = features_per_node
        self.min_observations_in_leaf_node = min_observations_in_leaf_node
        self.memory_saving_mode = memory_saving_mode
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.random_state = random_state

    def _check_params(self):
        if self.split_method not in ("inexact", "exact"):
            raise ValueError('Parameter "split_method" must be ' '"inexact" or "exact".')
        if (
            not isinstance(self.max_iterations, numbers.Integral)
            or self.max_iterations <= 0
        ):
            raise ValueError(
                'Parameter "max_iterations" must be ' "non-zero positive integer value."
            )
        if (
            not isinstance(self.max_tree_depth, numbers.Integral)
            or self.max_tree_depth < 0
        ):
            raise ValueError(
                'Parameter "max_tree_depth" must be ' "positive integer value or zero."
            )
        if self.shrinkage < 0 or self.shrinkage >= 1:
            raise ValueError(
                'Parameter "shrinkage" must be ' "more or equal to 0 and less than 1."
            )
        if self.min_split_loss < 0:
            raise ValueError(
                'Parameter "min_split_loss" must be ' "more or equal to zero."
            )
        if self.reg_lambda < 0:
            raise ValueError('Parameter "reg_lambda" must be ' "more or equal to zero.")
        if (
            self.observations_per_tree_fraction <= 0
            or self.observations_per_tree_fraction > 1
        ):
            raise ValueError(
                'Parameter "observations_per_tree_fraction" must be '
                "more than 0 and less or equal to 1."
            )
        if (
            not isinstance(self.features_per_node, numbers.Integral)
            or self.features_per_node < 0
        ):
            raise ValueError(
                'Parameter "features_per_node" must be ' "positive integer value or zero."
            )
        if (
            not isinstance(self.min_observations_in_leaf_node, numbers.Integral)
            or self.min_observations_in_leaf_node <= 0
        ):
            raise ValueError(
                'Parameter "min_observations_in_leaf_node" must be '
                "non-zero positive integer value."
            )
        if not (isinstance(self.memory_saving_mode, bool)):
            raise ValueError('Parameter "memory_saving_mode" must be ' "boolean value.")
        if not isinstance(self.max_bins, numbers.Integral) or self.max_bins <= 0:
            raise ValueError(
                'Parameter "max_bins" must be ' "non-zero positive integer value."
            )
        if not isinstance(self.min_bin_size, numbers.Integral) or self.min_bin_size <= 0:
            raise ValueError(
                'Parameter "min_bin_size" must be ' "non-zero positive integer value."
            )

    allow_nan_ = False

    def _more_tags(self):
        return {"allow_nan": self.allow_nan_}

    if sklearn_check_version("1.6"):

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.input_tags.allow_nan = self.allow_nan_
            return tags


@control_n_jobs(decorated_methods=["fit", "predict"])
class GBTDAALClassifier(ClassifierMixin, GBTDAALBase):
    def fit(self, X, y):
        # Check the algorithm parameters
        self._check_params()

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=False, dtype=[np.float64, np.float32])

        check_classification_targets(y)

        # Encode labels
        le = preprocessing.LabelEncoder()
        le.fit(y)
        self.classes_ = le.classes_
        y_ = le.transform(y)

        # Convert to 2d array
        y_ = y_.reshape((-1, 1))

        self.n_outputs_ = y_.shape[1]

        self.n_classes_ = len(self.classes_)

        self.n_features_in_ = X.shape[1]

        # Classifier can't train when only one class is present.
        # Trivial case
        if self.n_classes_ == 1:
            return self

        # Get random seed
        rs_ = check_random_state(self.random_state)
        seed_ = rs_.randint(0, np.iinfo("i").max)

        # Define type of data
        fptype = getFPType(X)

        # Fit the model
        train_algo = d4p.gbt_classification_training(
            fptype=fptype,
            nClasses=self.n_classes_,
            splitMethod=self.split_method,
            maxIterations=self.max_iterations,
            maxTreeDepth=self.max_tree_depth,
            shrinkage=self.shrinkage,
            minSplitLoss=self.min_split_loss,
            lambda_=self.reg_lambda,
            observationsPerTreeFraction=self.observations_per_tree_fraction,
            featuresPerNode=self.features_per_node,
            minObservationsInLeafNode=self.min_observations_in_leaf_node,
            memorySavingMode=self.memory_saving_mode,
            maxBins=self.max_bins,
            minBinSize=self.min_bin_size,
            engine=d4p.engines_mcg59(seed=seed_),
        )
        train_result = train_algo.compute(X, y_)

        # Store the model
        self.daal_model_ = train_result.model

        # Return the classifier
        return self

    def _predict(
        self, X, resultsToEvaluate, pred_contribs=False, pred_interactions=False
    ):
        # Check is fit had been called
        check_is_fitted(self, ["n_features_in_", "n_classes_"])

        # Input validation
        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            ensure_all_finite="allow-nan" if self.allow_nan_ else True,
            reset=False,
        )

        # Trivial case
        if self.n_classes_ == 1:
            return np.full(X.shape[0], self.classes_[0])

        fptype = getFPType(X)
        predict_result = self._predict_classification(
            X, fptype, resultsToEvaluate, pred_contribs, pred_interactions
        )

        if resultsToEvaluate == "computeClassLabels" and not (
            pred_contribs or pred_interactions
        ):
            # Decode labels
            le = preprocessing.LabelEncoder()
            le.classes_ = self.classes_
            return le.inverse_transform(predict_result)
        return predict_result

    def predict(self, X, pred_contribs=False, pred_interactions=False):
        return self._predict(X, "computeClassLabels", pred_contribs, pred_interactions)

    def predict_proba(self, X):
        return self._predict(X, "computeClassProbabilities")

    def predict_log_proba(self, X):
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return np.log(proba)

        for k in range(self.n_outputs_):
            proba[k] = np.log(proba[k])

        return proba

    @staticmethod
    def convert_model(model):
        gbm = GBTDAALClassifier()
        gbm._convert_model(model)

        gbm.classes_ = model.classes_
        gbm.allow_nan_ = True
        return gbm


@control_n_jobs(decorated_methods=["fit", "predict"])
class GBTDAALRegressor(RegressorMixin, GBTDAALBase):
    def fit(self, X, y):
        # Check the algorithm parameters
        self._check_params()

        # Check that X and y have correct shape
        X, y = check_X_y(X, y, y_numeric=True, dtype=[np.float64, np.float32])

        # Convert to 2d array
        y_ = y.reshape((-1, 1))

        self.n_features_in_ = X.shape[1]

        # Get random seed
        rs_ = check_random_state(self.random_state)
        seed_ = rs_.randint(0, np.iinfo("i").max)

        # Define type of data
        fptype = getFPType(X)

        # Fit the model
        train_algo = d4p.gbt_regression_training(
            fptype=fptype,
            splitMethod=self.split_method,
            maxIterations=self.max_iterations,
            maxTreeDepth=self.max_tree_depth,
            shrinkage=self.shrinkage,
            minSplitLoss=self.min_split_loss,
            lambda_=self.reg_lambda,
            observationsPerTreeFraction=self.observations_per_tree_fraction,
            featuresPerNode=self.features_per_node,
            minObservationsInLeafNode=self.min_observations_in_leaf_node,
            memorySavingMode=self.memory_saving_mode,
            maxBins=self.max_bins,
            minBinSize=self.min_bin_size,
            engine=d4p.engines_mcg59(seed=seed_),
        )
        train_result = train_algo.compute(X, y_)

        # Store the model
        self.daal_model_ = train_result.model

        # Return the classifier
        return self

    def predict(self, X, pred_contribs=False, pred_interactions=False):
        # Check is fit had been called
        check_is_fitted(self, ["n_features_in_"])

        # Input validation
        X = validate_data(
            self,
            X,
            dtype=[np.float64, np.float32],
            ensure_all_finite="allow-nan" if self.allow_nan_ else True,
            reset=False,
        )

        fptype = getFPType(X)
        return self._predict_regression(X, fptype, pred_contribs, pred_interactions)

    @staticmethod
    def convert_model(model):
        gbm = GBTDAALRegressor()
        gbm._convert_model(model)

        gbm.allow_nan_ = True
        return gbm

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

# daal4py AdaBoost (Adaptive Boosting) scikit-learn-compatible estimator class

import numbers

import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import daal4py as d4p
from daal4py.sklearn._utils import sklearn_check_version

from .._n_jobs_support import control_n_jobs
from .._utils import getFPType
from ..utils.validation import validate_data


@control_n_jobs(decorated_methods=["fit", "predict"])
class AdaBoostClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        split_criterion="gini",
        max_tree_depth=1,
        min_observations_in_leaf_node=1,
        max_iterations=100,
        learning_rate=1.0,
        accuracy_threshold=0.01,
    ):
        self.split_criterion = split_criterion
        self.max_tree_depth = max_tree_depth
        self.min_observations_in_leaf_node = min_observations_in_leaf_node
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.accuracy_threshold = accuracy_threshold

    def fit(self, X, y):
        if self.split_criterion not in ("gini", "infoGain"):
            raise ValueError(
                'Parameter "split_criterion" must be ' '"gini" or "infoGain".'
            )
        if (
            not isinstance(self.max_tree_depth, numbers.Integral)
            or self.max_tree_depth < 0
        ):
            raise ValueError(
                'Parameter "max_tree_depth" must be ' "positive integer value or zero."
            )
        if (
            not isinstance(self.min_observations_in_leaf_node, numbers.Integral)
            or self.min_observations_in_leaf_node <= 0
        ):
            raise ValueError(
                'Parameter "min_observations_in_leaf_node" must be '
                "non-zero positive integer value."
            )
        if (
            not isinstance(self.max_iterations, numbers.Integral)
            or self.max_iterations <= 0
        ):
            raise ValueError(
                'Parameter "max_iterations" must be ' "non-zero positive integer value."
            )
        if self.learning_rate <= 0:
            raise ValueError(
                'Parameter "learning_rate" must be ' "non-zero positive value."
            )
        # it is not clear why it is so but we will get error from
        # oneAPI Data Analytics
        # Library otherwise
        if self.accuracy_threshold < 0 and self.accuracy_threshold >= 1:
            raise ValueError(
                'Parameter "accuracy_threshold" must be '
                "more or equal to 0 and less than 1."
            )

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

        self.n_classes_ = len(self.classes_)

        self.n_features_in_ = X.shape[1]

        # Classifier can't train when only one class is present.
        # Trivial case
        if self.n_classes_ == 1:
            return self

        # Define type of data
        fptype = getFPType(X)

        # Fit the model
        tr = d4p.decision_tree_classification_training(
            fptype=fptype,
            nClasses=self.n_classes_,
            # this parameter is strict upper bound in DAAL
            maxTreeDepth=self.max_tree_depth + 1,
            minObservationsInLeafNodes=self.min_observations_in_leaf_node,
            splitCriterion=self.split_criterion,
            pruning="none",
        )

        pr = d4p.decision_tree_classification_prediction(
            fptype=fptype, nClasses=self.n_classes_
        )

        train_algo = d4p.adaboost_training(
            fptype=fptype,
            nClasses=self.n_classes_,
            weakLearnerTraining=tr,
            weakLearnerPrediction=pr,
            maxIterations=self.max_iterations,
            learningRate=self.learning_rate,
            accuracyThreshold=self.accuracy_threshold,
        )

        train_result = train_algo.compute(X, y_)

        # Store the model
        self.daal_model_ = train_result.model

        # Return the classifier
        return self

    def predict(self, X):
        check_is_fitted(self)

        # Input validation
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)

        # Trivial case
        if self.n_classes_ == 1:
            return np.full(X.shape[0], self.classes_[0])

        if not hasattr(self, "daal_model_"):
            raise ValueError(
                (
                    "The class {} instance does not have 'daal_model_' attribute set. "
                    "Call 'fit' with appropriate arguments before using this method."
                ).format(type(self).__name__)
            )

        # Define type of data
        fptype = getFPType(X)

        pr = d4p.decision_tree_classification_prediction(
            fptype=fptype, nClasses=self.n_classes_
        )

        # Prediction
        predict_algo = d4p.adaboost_prediction(
            fptype=fptype, nClasses=self.n_classes_, weakLearnerPrediction=pr
        )
        predict_result = predict_algo.compute(X, self.daal_model_)

        prediction = predict_result.prediction

        # in binary classification labels "-1, 1" are returned but "0, 1" are expected
        if self.n_classes_ == 2:
            prediction[prediction == -1] = 0

        # Decode labels
        le = preprocessing.LabelEncoder()
        le.classes_ = self.classes_
        return le.inverse_transform(prediction.ravel().astype(np.int64, copy=False))

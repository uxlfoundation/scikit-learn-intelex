# ==============================================================================
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
# ==============================================================================

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from sklearnex.decomposition import PCA
from sklearnex.ensemble import RandomForestClassifier
from sklearnex.linear_model import LinearRegression
from sklearnex.preview.covariance import EmpiricalCovariance

# Table of estimators to test hyperparameter reset functionality.
# Each row contains the following elements:
#
# [estimator, estimator_type, operation, hyperparameter_name, non_default_value]
#
# estimator:            the estimator class to test. For example, EmpiricalCovariance.
# estimator_type:       the type of operation to get.
#                       Possible values: "compute", "regression", "classification".
# operation:            the argument used in get_hyperparameters() and reset_hyperparameters()
#                       methods of the estimator.
# hyperparameter_name:  the name of the hyperparameter to test.
# non_default_value:    the value to set for the hyperparameter before resetting it.
#                       This value should be different from the default value of the hyperparameter.
test_estimators = [
    [EmpiricalCovariance, "compute", "fit", "cpu_macro_block", 10],
    [EmpiricalCovariance, "compute", "fit", "cpu_grain_size", 2],
    [LinearRegression, "regression", "fit", "cpu_macro_block", 10],
    [PCA, "compute", "fit", "cpu_macro_block", 10],
    [RandomForestClassifier, "classification", "predict", "block_size", 8],
]


def call_estimator(estimator_object, estimator_type, op, X, y=None):
    if estimator_type == "compute":
        return estimator_object.fit(X)
    elif estimator_type == "regression" or estimator_type == "classification":
        result = estimator_object.fit(X, y)
        if op == "predict":
            return estimator_object.predict(X)
        else:
            return result
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


def test_reset_hyperparameters():
    for estimator, estimator_type, op, param_name, non_default_value in test_estimators:

        if estimator_type == "compute":
            X = np.random.rand(100, 5)
            y = None
        elif estimator_type == "regression":
            X, y = make_regression(n_features=5, n_informative=5, random_state=777)
        elif estimator_type == "classification":
            X, y = make_classification(n_features=5, random_state=777)

        # Create an instance of the estimator
        est_object = estimator()

        # Fit the model to the data with the default hyperparameters
        call_estimator(est_object, estimator_type, op, X, y)

        # Get the hyperparameters before resetting
        hparams_before = estimator.get_hyperparameters(op)
        default_hparam_value = getattr(hparams_before, param_name)

        # Fit the model to the data with non-default hyperparameters
        setattr(hparams_before, param_name, non_default_value)
        call_estimator(est_object, estimator_type, op, X, y)

        # Check if the hyperparameters have been set to non-default values
        assert getattr(hparams_before, param_name) == non_default_value
        assert getattr(estimator.get_hyperparameters(op), param_name) == non_default_value

        # Reset the hyperparameters
        estimator.reset_hyperparameters(op)
        call_estimator(est_object, estimator_type, op, X, y)

        # Check if the hyperparameters have been reset to default values
        assert (
            getattr(estimator.get_hyperparameters(op), param_name) == default_hparam_value
        )

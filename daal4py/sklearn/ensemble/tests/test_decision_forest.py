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

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier as ScikitRandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as ScikitRandomForestRegressor
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from daal4py.sklearn.ensemble import RandomForestClassifier as DaalRandomForestClassifier
from daal4py.sklearn.ensemble import RandomForestRegressor as DaalRandomForestRegressor

ACCURACY_RATIO = 0.95 if daal_check_version((2021, "P", 400)) else 0.85
MSE_RATIO = 1.07
LOG_LOSS_RATIO = 1.55
ROC_AUC_RATIO = 0.96
RNG = np.random.RandomState(0)
IRIS = load_iris()


def _compare_with_sklearn_classifier_iris(
    n_estimators=99, class_weight=None, sample_weight=None, description=""
):
    x_train, x_test, y_train, y_test = train_test_split(
        IRIS.data, IRIS.target, test_size=0.33, random_state=31
    )
    # models
    scikit_model = ScikitRandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_depth=None,
        random_state=777,
    )
    daal4py_model = DaalRandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_depth=None,
        random_state=777,
    )
    # training
    scikit_model.fit(x_train, y_train, sample_weight=sample_weight)
    daal4py_model.fit(x_train, y_train, sample_weight=sample_weight)
    # predict
    scikit_predict = scikit_model.predict(x_test)
    daal4py_predict = daal4py_model.predict(x_test)
    # accuracy
    scikit_accuracy = accuracy_score(scikit_predict, y_test)
    daal4py_accuracy = accuracy_score(daal4py_predict, y_test)
    ratio = daal4py_accuracy / scikit_accuracy
    reason = (
        description
        + f"scikit_accuracy={scikit_accuracy}, daal4py_accuracy={daal4py_accuracy}"
    )
    assert ratio >= ACCURACY_RATIO, reason

    # predict_proba
    scikit_predict_proba = scikit_model.predict_proba(x_test)
    daal4py_predict_proba = daal4py_model.predict_proba(x_test)
    # log_loss
    scikit_log_loss = log_loss(y_test, scikit_predict_proba)
    daal4py_log_loss = log_loss(y_test, daal4py_predict_proba)
    ratio = daal4py_log_loss / scikit_log_loss

    reason = (
        description
        + f"scikit_log_loss={scikit_log_loss}, daal4py_log_loss={daal4py_log_loss}"
    )
    assert ratio <= LOG_LOSS_RATIO, reason

    # ROC AUC
    scikit_roc_auc = roc_auc_score(y_test, scikit_predict_proba, multi_class="ovr")
    daal4py_roc_auc = roc_auc_score(y_test, daal4py_predict_proba, multi_class="ovr")
    ratio = daal4py_roc_auc / scikit_roc_auc

    reason = (
        description
        + f"scikit_roc_auc={scikit_roc_auc}, daal4py_roc_auc={daal4py_roc_auc}"
    )
    assert ratio >= ROC_AUC_RATIO, reason


CLASS_WEIGHTS_IRIS = [
    {0: 0, 1: 1, 2: 1},
    {0: 1, 1: 2, 2: 3},
    {0: 10, 1: 5, 2: 4},
    {
        0: RNG.uniform(1, 50),
        1: RNG.uniform(1, 50),
        2: RNG.uniform(1, 50),
    },
    {
        0: RNG.uniform(1, 1000),
        1: RNG.uniform(1, 1000),
        2: RNG.uniform(1, 1000),
    },
    {
        0: RNG.uniform(1, 10),
        1: RNG.uniform(50, 100),
        2: RNG.uniform(1, 100),
    },
    {0: 50, 1: 50, 2: 50},
    "balanced",
]


@pytest.mark.parametrize("class_weight", CLASS_WEIGHTS_IRIS)
def test_classifier_class_weight_iris(class_weight):
    _compare_with_sklearn_classifier_iris(
        class_weight=class_weight, description="Classifier class weight: "
    )


SAMPLE_WEIGHTS_IRIS = [
    (np.full_like(range(100), 1), "Only 1"),
    (np.full_like(range(100), 50), "Only 50"),
    (RNG.rand(100), "Uniform distribution"),
    (RNG.normal(1000, 10, 100), "Gaussian distribution"),
    (RNG.poisson(lam=10, size=100), "Poisson distribution"),
    (RNG.rayleigh(scale=1, size=100), "Rayleigh distribution"),
]


@pytest.mark.parametrize("sample_weight", SAMPLE_WEIGHTS_IRIS)
def test_classifier_sample_weight_iris(sample_weight):
    sample_weight, description = sample_weight
    _compare_with_sklearn_classifier_iris(
        sample_weight=sample_weight,
        description=f"Classifier sample_weight_type={description}: ",
    )


N_ESTIMATORS_IRIS = [
    1000,
    8000,
]


@pytest.mark.parametrize("n_estimators", N_ESTIMATORS_IRIS)
def test_classifier_big_estimators_iris(n_estimators):
    _compare_with_sklearn_classifier_iris(
        n_estimators=n_estimators, description=f"Classifier n_estimators={n_estimators}: "
    )


def _compare_with_sklearn_mse_regressor_iris(
    n_estimators=99, sample_weight=None, description=""
):
    x_train, x_test, y_train, y_test = train_test_split(
        IRIS.data, IRIS.target, test_size=0.33, random_state=31
    )

    scikit_model = ScikitRandomForestRegressor(
        n_estimators=n_estimators, max_depth=None, random_state=777
    )
    daal4py_model = DaalRandomForestRegressor(
        n_estimators=n_estimators, max_depth=None, random_state=777
    )

    scikit_predict = scikit_model.fit(
        x_train, y_train, sample_weight=sample_weight
    ).predict(x_test)
    daal4py_predict = daal4py_model.fit(
        x_train, y_train, sample_weight=sample_weight
    ).predict(x_test)

    scikit_mse = mean_squared_error(scikit_predict, y_test)
    daal4py_mse = mean_squared_error(daal4py_predict, y_test)

    ratio = daal4py_mse / scikit_mse
    reason = description + f"scikit_mse={scikit_mse}, daal4py_mse={daal4py_mse}"
    assert ratio <= MSE_RATIO, reason


@pytest.mark.skipif(
    sklearn_check_version("1.9"), reason="Changes in RF methodology in sklearn1.9"
)
@pytest.mark.parametrize("weight", SAMPLE_WEIGHTS_IRIS)
def test_mse_regressor_sample_weight_iris(weight):
    sample_weight, description = weight
    _compare_with_sklearn_mse_regressor_iris(
        sample_weight=sample_weight,
        description=f"Regression sample weights: sample_weight_type={description}: ",
    )


@pytest.mark.parametrize("n_estimators", N_ESTIMATORS_IRIS)
def test_mse_regressor_big_estimators_iris(n_estimators):
    _compare_with_sklearn_mse_regressor_iris(
        n_estimators=n_estimators,
        description=f"Regression: n_estimators={n_estimators}: ",
    )

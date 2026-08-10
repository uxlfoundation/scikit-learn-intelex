# ==============================================================================
# Copyright 2021 Intel Corporation
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
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

MODELS_INFO = [
    {
        "model": KNeighborsClassifier(algorithm="brute"),
        "methods": ["kneighbors", "predict", "predict_proba", "score"],
        "dataset": "classifier",
    },
    {
        "model": KNeighborsRegressor(algorithm="brute"),
        "methods": ["kneighbors", "predict", "score"],
        "dataset": "regression",
    },
    {
        "model": NearestNeighbors(algorithm="brute"),
        "methods": ["kneighbors"],
        "dataset": "blobs",
    },
    {
        "model": ElasticNet(),
        "methods": ["predict", "score"],
        "dataset": "regression",
    },
    {
        "model": Lasso(),
        "methods": ["predict", "score"],
        "dataset": "regression",
    },
    {
        "model": LogisticRegression(
            max_iter=100,
            **({} if sklearn_check_version("1.8") else {"multi_class": "multinomial"})
        ),
        "methods": [
            "decision_function",
            "predict",
            "predict_proba",
            "predict_log_proba",
            "score",
        ],
        "dataset": "classifier",
    },
    {
        "model": LogisticRegressionCV(
            max_iter=100,
            **({} if sklearn_check_version("1.8") else {"multi_class": "multinomial"})
        ),
        "methods": [
            "decision_function",
            "predict",
            "predict_proba",
            "predict_log_proba",
            "score",
        ],
        "dataset": "classifier",
    },
]

TYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]

TO_SKIP = [
    # --------------- NO INFO ---------------
    r"LogisticRegression .*decision_function",
    r"LogisticRegressionCV .*score",
    r"LogisticRegressionCV .*decision_function",
    r"LogisticRegressionCV .*score",
    # --------------- Scikit ---------------
    r"pairwise_distances .*pairwise_distances",  # except float64
    (
        r"roc_auc_score .*roc_auc_score"
        if not daal_check_version((2021, "P", 200))
        else None
    ),
]

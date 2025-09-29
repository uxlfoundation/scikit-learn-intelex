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
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
)
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, NearestNeighbors
from sklearn.svm import SVC

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
        "model": DBSCAN(),
        "methods": ["fit_predict"],
        "dataset": "blobs",
    },
    {
        "model": SVC(probability=True),
        "methods": ["decision_function", "predict", "predict_proba", "score"],
        "dataset": "classifier",
    },
    {
        "model": KMeans(),
        "methods": ["fit_predict", "fit_transform", "transform", "predict", "score"],
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
        "model": PCA(),
        "methods": ["fit_transform", "transform", "score"],
        "dataset": "classifier",
    },
    {
        "model": RandomForestClassifier(n_estimators=10),
        "methods": ["predict", "predict_proba", "predict_log_proba", "score"],
        "dataset": "classifier",
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
        "model": LogisticRegressionCV(max_iter=100),
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
        "model": RandomForestRegressor(n_estimators=10),
        "methods": ["predict", "score"],
        "dataset": "regression",
    },
    {
        "model": LinearRegression(),
        "methods": ["predict", "score"],
        "dataset": "regression",
    },
    {
        "model": Ridge(),
        "methods": ["predict", "score"],
        "dataset": "regression",
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
    r"KMeans .*transform",
    r"KMeans .*score",
    r"PCA .*score",
    r"LogisticRegression .*decision_function",
    r"LogisticRegressionCV .*decision_function",
    r"LogisticRegressionCV .*predict",
    r"LogisticRegressionCV .*predict_proba",
    r"LogisticRegressionCV .*predict_log_proba",
    r"LogisticRegressionCV .*score",
    # --------------- Scikit ---------------
    r"Ridge float16 predict",
    r"Ridge float16 score",
    r"RandomForestClassifier .*predict_proba",
    r"RandomForestClassifier .*predict_log_proba",
    r"pairwise_distances .*pairwise_distances",  # except float64
    (
        r"roc_auc_score .*roc_auc_score"
        if not daal_check_version((2021, "P", 200))
        else None
    ),
]

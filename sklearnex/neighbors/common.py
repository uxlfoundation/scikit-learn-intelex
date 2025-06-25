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

import warnings

import numpy as np
from scipy import sparse as sp
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._base import VALID_METRICS, KNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as _sklearn_NeighborsBase
from sklearn.neighbors._kd_tree import KDTree
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._utils import sklearn_check_version
from onedal.utils.validation import _check_array, _num_features, _num_samples

from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import get_namespace
from ..utils.validation import check_feature_names


class KNeighborsDispatchingBase(oneDALEstimator):
    def _fit_validation(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        check_feature_names(self, X, reset=True)
        if self.metric_params is not None and "p" in self.metric_params:
            if self.p is not None:
                warnings.warn(
                    "Parameter p is found in metric_params. "
                    "The corresponding parameter from __init__ "
                    "is ignored.",
                    SyntaxWarning,
                    stacklevel=2,
                )
            self.effective_metric_params_ = self.metric_params.copy()
            effective_p = self.metric_params["p"]
        else:
            self.effective_metric_params_ = {}
            effective_p = self.p

        self.effective_metric_params_["p"] = effective_p
        self.effective_metric_ = self.metric
        # For minkowski distance, use more efficient methods where available
        if self.metric == "minkowski":
            p = self.effective_metric_params_["p"]
            if p == 1:
                self.effective_metric_ = "manhattan"
            elif p == 2:
                self.effective_metric_ = "euclidean"
            elif p == np.inf:
                self.effective_metric_ = "chebyshev"

        if not isinstance(X, (KDTree, BallTree, _sklearn_NeighborsBase)):
            self._fit_X = _check_array(
                X, dtype=[np.float64, np.float32], accept_sparse=True
            )
            self.n_samples_fit_ = _num_samples(self._fit_X)
            self.n_features_in_ = _num_features(self._fit_X)

            if self.algorithm == "auto":
                # A tree approach is better for small number of neighbors or small
                # number of features, with KDTree generally faster when available
                is_n_neighbors_valid_for_brute = (
                    self.n_neighbors is not None
                    and self.n_neighbors >= self._fit_X.shape[0] // 2
                )
                if self._fit_X.shape[1] > 15 or is_n_neighbors_valid_for_brute:
                    self._fit_method = "brute"
                else:
                    if self.effective_metric_ in VALID_METRICS["kd_tree"]:
                        self._fit_method = "kd_tree"
                    elif (
                        callable(self.effective_metric_)
                        or self.effective_metric_ in VALID_METRICS["ball_tree"]
                    ):
                        self._fit_method = "ball_tree"
                    else:
                        self._fit_method = "brute"
            else:
                self._fit_method = self.algorithm

        if hasattr(self, "_onedal_estimator"):
            delattr(self, "_onedal_estimator")
        # To cover test case when we pass patched
        # estimator as an input for other estimator
        if isinstance(X, _sklearn_NeighborsBase):
            self._fit_X = X._fit_X
            self._tree = X._tree
            self._fit_method = X._fit_method
            self.n_samples_fit_ = X.n_samples_fit_
            self.n_features_in_ = X.n_features_in_
            if hasattr(X, "_onedal_estimator"):
                self.effective_metric_params_.pop("p")
                if self._fit_method == "ball_tree":
                    X._tree = BallTree(
                        X._fit_X,
                        self.leaf_size,
                        metric=self.effective_metric_,
                        **self.effective_metric_params_,
                    )
                elif self._fit_method == "kd_tree":
                    X._tree = KDTree(
                        X._fit_X,
                        self.leaf_size,
                        metric=self.effective_metric_,
                        **self.effective_metric_params_,
                    )
                elif self._fit_method == "brute":
                    X._tree = None
                else:
                    raise ValueError("algorithm = '%s' not recognized" % self.algorithm)

        elif isinstance(X, BallTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = "ball_tree"
            self.n_samples_fit_ = X.data.shape[0]
            self.n_features_in_ = X.data.shape[1]

        elif isinstance(X, KDTree):
            self._fit_X = X.data
            self._tree = X
            self._fit_method = "kd_tree"
            self.n_samples_fit_ = X.data.shape[0]
            self.n_features_in_ = X.data.shape[1]

    def _onedal_supported(self, device, method_name, *data):
        if method_name == "fit":
            self._fit_validation(data[0], data[1])

        class_name = self.__class__.__name__
        is_classifier = "Classifier" in class_name
        is_regressor = "Regressor" in class_name
        is_unsupervised = not (is_classifier or is_regressor)
        patching_status = PatchingConditionsChain(
            f"sklearn.neighbors.{class_name}.{method_name}"
        )
        if not patching_status.and_condition(
            "radius" not in method_name, "RadiusNeighbors not implemented in sklearnex"
        ):
            return patching_status

        if not patching_status.and_condition(
            not isinstance(data[0], (KDTree, BallTree, _sklearn_NeighborsBase)),
            f"Input type {type(data[0])} is not supported.",
        ):
            return patching_status

        if self._fit_method in ["auto", "ball_tree"]:
            condition = (
                self.n_neighbors is not None
                and self.n_neighbors >= self.n_samples_fit_ // 2
            )
            if self.n_features_in_ > 15 or condition:
                result_method = "brute"
            else:
                if self.effective_metric_ in ["euclidean"]:
                    result_method = "kd_tree"
                else:
                    result_method = "brute"
        else:
            result_method = self._fit_method

        p_less_than_one = (
            "p" in self.effective_metric_params_.keys()
            and self.effective_metric_params_["p"] < 1
        )
        if not patching_status.and_condition(
            not p_less_than_one, '"p" metric parameter is less than 1'
        ):
            return patching_status

        if not patching_status.and_condition(
            not sp.issparse(data[0]), "Sparse input is not supported."
        ):
            return patching_status

        if not is_unsupervised:
            is_valid_weights = self.weights in ["uniform", "distance"]
            if is_classifier:
                class_count = 1
            is_single_output = False
            y = None
            # To check multioutput, might be overhead
            if len(data) > 1:
                y = np.asarray(data[1])
                if is_classifier:
                    class_count = len(np.unique(y))
            if hasattr(self, "_onedal_estimator"):
                y = self._onedal_estimator._y
            if y is not None and hasattr(y, "ndim") and hasattr(y, "shape"):
                is_single_output = y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1

        # TODO: add native support for these metric names
        metrics_map = {"manhattan": ["l1", "cityblock"], "euclidean": ["l2"]}
        for origin, aliases in metrics_map.items():
            if self.effective_metric_ in aliases:
                self.effective_metric_ = origin
                break
        if self.effective_metric_ == "manhattan":
            self.effective_metric_params_["p"] = 1
        elif self.effective_metric_ == "euclidean":
            self.effective_metric_params_["p"] = 2

        onedal_brute_metrics = [
            "manhattan",
            "minkowski",
            "euclidean",
            "chebyshev",
            "cosine",
        ]
        onedal_kdtree_metrics = ["euclidean"]
        is_valid_for_brute = (
            result_method == "brute" and self.effective_metric_ in onedal_brute_metrics
        )
        is_valid_for_kd_tree = (
            result_method == "kd_tree" and self.effective_metric_ in onedal_kdtree_metrics
        )
        if result_method == "kd_tree":
            if not patching_status.and_condition(
                device != "gpu", '"kd_tree" method is not supported on GPU.'
            ):
                return patching_status

        if not patching_status.and_condition(
            is_valid_for_kd_tree or is_valid_for_brute,
            f"{result_method} with {self.effective_metric_} metric is not supported.",
        ):
            return patching_status
        if not is_unsupervised:
            if not patching_status.and_conditions(
                [
                    (is_single_output, "Only single output is supported."),
                    (
                        is_valid_weights,
                        f'"{type(self.weights)}" weights type is not supported.',
                    ),
                ]
            ):
                return patching_status
        if method_name == "fit":
            if is_classifier:
                patching_status.and_condition(
                    class_count >= 2, "One-class case is not supported."
                )
            return patching_status
        if method_name in ["predict", "predict_proba", "kneighbors", "score"]:
            patching_status.and_condition(
                hasattr(self, "_onedal_estimator"), "oneDAL model was not trained."
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {class_name}")

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported("gpu", method_name, *data)

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported("cpu", method_name, *data)

    def kneighbors_graph(self, X=None, n_neighbors=None, mode="connectivity"):
        check_is_fitted(self)
        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # check the input only in self.kneighbors

        # construct CSR matrix representation of the k-NN graph
        if mode == "connectivity":
            A_ind = self.kneighbors(X, n_neighbors, return_distance=False)
            xp, _ = get_namespace(A_ind)
            n_queries = A_ind.shape[0]
            A_data = xp.ones(n_queries * n_neighbors)

        elif mode == "distance":
            A_data, A_ind = self.kneighbors(X, n_neighbors, return_distance=True)
            xp, _ = get_namespace(A_ind)
            A_data = xp.reshape(A_data, (-1,))

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                f'or "distance" but got "{mode}" instead'
            )

        n_queries = A_ind.shape[0]
        n_samples_fit = self.n_samples_fit_
        n_nonzero = n_queries * n_neighbors
        A_indptr = xp.arange(0, n_nonzero + 1, n_neighbors)

        kneighbors_graph = sp.csr_matrix(
            (A_data, xp.reshape(A_ind, (-1,)), A_indptr), shape=(n_queries, n_samples_fit)
        )

        return kneighbors_graph

    kneighbors_graph.__doc__ = KNeighborsMixin.kneighbors_graph.__doc__

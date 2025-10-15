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
from numbers import Integral

import numpy as np
from scipy import sparse as sp
from sklearn.neighbors._ball_tree import BallTree
from sklearn.neighbors._base import VALID_METRICS, KNeighborsMixin
from sklearn.neighbors._base import NeighborsBase as _sklearn_NeighborsBase
from sklearn.neighbors._kd_tree import KDTree
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version

from ..utils.validation import validate_data
from onedal._device_offload import _transfer_to_host
from onedal.utils.validation import (
    _check_array,
    _check_classification_targets,
    _check_X_y,
    _column_or_1d,
    _num_features,
    _num_samples,
)

from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import get_namespace
from ..utils.validation import check_feature_names


class KNeighborsDispatchingBase(oneDALEstimator):
    def _parse_auto_method(self, method, n_samples, n_features):
        result_method = method

        if method in ["auto", "ball_tree"]:
            condition = (
                self.n_neighbors is not None and self.n_neighbors >= n_samples // 2
            )
            if self.metric == "precomputed" or n_features > 15 or condition:
                result_method = "brute"
            else:
                if self.metric == "euclidean":
                    result_method = "kd_tree"
                else:
                    result_method = "brute"

        return result_method

    # def _validate_data(
    #     self, X, y=None, reset=True, validate_separately=None, **check_params
    # ):
    #     if y is None:
    #         if getattr(self, "requires_y", False):
    #             raise ValueError(
    #                 f"This {self.__class__.__name__} estimator "
    #                 f"requires y to be passed, but the target y is None."
    #             )
    #         X = _check_array(X, **check_params)
    #         out = X, y
    #     else:
    #         if validate_separately:
    #             # We need this because some estimators validate X and y
    #             # separately, and in general, separately calling _check_array()
    #             # on X and y isn't equivalent to just calling _check_X_y()
    #             # :(
    #             check_X_params, check_y_params = validate_separately
    #             X = _check_array(X, **check_X_params)
    #             y = _check_array(y, **check_y_params)
    #         else:
    #             X, y = _check_X_y(X, y, **check_params)
    #         out = X, y

    #     if check_params.get("ensure_2d", True):
    #         from onedal.utils.validation import _check_n_features

    #         _check_n_features(self, X, reset=reset)

    #     return out

    def _get_weights(self, dist, weights):
        if weights in (None, "uniform"):
            return None
        if weights == "distance":
            # if user attempts to classify a point that was zero distance from one
            # or more training points, those training points are weighted as 1.0
            # and the other points as 0.0
            if dist.dtype is np.dtype(object):
                for point_dist_i, point_dist in enumerate(dist):
                    # check if point_dist is iterable
                    # (ex: RadiusNeighborClassifier.predict may set an element of
                    # dist to 1e-6 to represent an 'outlier')
                    if hasattr(point_dist, "__contains__") and 0.0 in point_dist:
                        dist[point_dist_i] = point_dist == 0.0
                    else:
                        dist[point_dist_i] = 1.0 / point_dist
            else:
                with np.errstate(divide="ignore"):
                    dist = 1.0 / dist
                inf_mask = np.isinf(dist)
                inf_row = np.any(inf_mask, axis=1)
                dist[inf_row] = inf_mask[inf_row]
            return dist
        elif callable(weights):
            return weights(dist)
        else:
            raise ValueError(
                "weights not recognized: should be 'uniform', "
                "'distance', or a callable function"
            )

    def _validate_targets(self, y, dtype):
        arr = _column_or_1d(y, warn=True)

        try:
            return arr.astype(dtype, copy=False)
        except ValueError:
            return arr

    def _validate_n_neighbors(self, n_neighbors):
        if n_neighbors is not None:
            if n_neighbors <= 0:
                raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
            if not isinstance(n_neighbors, Integral):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" % type(n_neighbors)
                )

    def _validate_n_classes(self):
        """Validate that the classifier has at least 2 classes."""
        length = 0 if self.classes_ is None else len(self.classes_)
        if length < 2:
            raise ValueError(
                f"The number of classes has to be greater than one; got {length}"
            )

    def _validate_feature_count(self, X, method_name=""):
        n_features = getattr(self, "n_features_in_", None)
        shape = getattr(X, "shape", None)
        if n_features and shape and len(shape) > 1 and shape[1] != n_features:
            raise ValueError(
                (
                    f"X has {X.shape[1]} features, "
                    f"but {method_name} is expecting "
                    f"{n_features} features as input"
                )
            )

    def _validate_kneighbors_bounds(self, n_neighbors, query_is_train, X):
        n_samples_fit = self.n_samples_fit_
        if n_neighbors > n_samples_fit:
            if query_is_train:
                n_neighbors -= 1  # ok to modify inplace because an error is raised
                inequality_str = "n_neighbors < n_samples_fit"
            else:
                inequality_str = "n_neighbors <= n_samples_fit"
            raise ValueError(
                f"Expected {inequality_str}, but "
                f"n_neighbors = {n_neighbors}, n_samples_fit = {n_samples_fit}, "
                f"n_samples = {X.shape[0]}"  # include n_samples for common tests
            )

    def _kneighbors_validation(self, X, n_neighbors):
        """Shared validation for kneighbors method called from sklearnex layer.
        
        Validates:
        - Feature count matches training data if X is provided
        - n_neighbors is within valid bounds if provided
        """
        # Validate feature count if X is provided
        if X is not None:
            self._validate_feature_count(X)
        
        # Validate n_neighbors bounds if provided
        if n_neighbors is not None:
            # Determine if query is the training set
            query_is_train = X is None or (hasattr(self, '_fit_X') and X is self._fit_X)
            self._validate_kneighbors_bounds(n_neighbors, query_is_train, X if X is not None else self._fit_X)

    def _prepare_kneighbors_inputs(self, X, n_neighbors):
        """Prepare inputs for kneighbors call to onedal backend.
        
        Handles query_is_train case: when X=None, sets X to training data and adds +1 to n_neighbors.
        
        Args:
            X: Query data or None
            n_neighbors: Number of neighbors or None
            
        Returns:
            Tuple of (X, n_neighbors, query_is_train)
            - X: Processed query data (self._fit_X if original X was None)
            - n_neighbors: Adjusted n_neighbors (includes +1 if query_is_train)
            - query_is_train: Boolean flag indicating if original X was None
        """
        query_is_train = X is None
        
        if X is not None:
            # Validate and convert X (pandas to numpy if needed)
            X = validate_data(
                self, X, dtype=[np.float64, np.float32], accept_sparse="csr", reset=False
            )
        else:
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            if n_neighbors is None:
                n_neighbors = self.n_neighbors
            n_neighbors += 1
        
        return X, n_neighbors, query_is_train

    def _kneighbors_post_processing(self, X, n_neighbors, return_distance, result, query_is_train):
        """Shared post-processing for kneighbors results.
        
        Following PCA pattern: all post-processing in sklearnex, onedal returns raw results.
        
        Handles:
        - query_is_train case (X=None): removes self from results
        - kd_tree sorting: sorts results by distance
        
        Args:
            X: Query data (self._fit_X if query_is_train)
            n_neighbors: Number of neighbors (already includes +1 if query_is_train)
            return_distance: Whether distances are included in result
            result: Raw result from onedal backend (distances, indices) or just indices
            query_is_train: Boolean indicating if original X was None
        
        Returns:
            Post-processed result in same format as input result
        """
        # POST-PROCESSING: kd_tree sorting (moved from onedal)
        if self._fit_method == "kd_tree":
            if return_distance:
                distances, indices = result
                for i in range(distances.shape[0]):
                    seq = distances[i].argsort()
                    indices[i] = indices[i][seq]
                    distances[i] = distances[i][seq]
                result = distances, indices
            else:
                indices = result
                # For indices-only, we still need to sort but we don't have distances
                # In this case, indices should already be sorted by onedal
                pass
        
        # POST-PROCESSING: Remove self from results when query_is_train (moved from onedal)
        if query_is_train:
            if return_distance:
                neigh_dist, neigh_ind = result
            else:
                neigh_ind = result
            
            # X is self._fit_X in query_is_train case (set by caller)
            n_queries, _ = X.shape
            sample_range = np.arange(n_queries)[:, None]
            sample_mask = neigh_ind != sample_range
            
            # Corner case: When the number of duplicates are more
            # than the number of neighbors, the first NN will not
            # be the sample, but a duplicate.
            # In that case mask the first duplicate.
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False
            
            neigh_ind = np.reshape(neigh_ind[sample_mask], (n_queries, n_neighbors - 1))
            
            if return_distance:
                neigh_dist = np.reshape(neigh_dist[sample_mask], (n_queries, n_neighbors - 1))
                result = neigh_dist, neigh_ind
            else:
                result = neigh_ind
        
        return result

    def _process_classification_targets(self, y):
        """Process classification targets and set class-related attributes.
        
        Note: y should already be converted to numpy array via validate_data before calling this.
        """
        import sys
        print(f"DEBUG _process_classification_targets: y type={type(y)}, y shape={getattr(y, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        
        # y should already be numpy array from validate_data
        y = np.asarray(y)

        # Handle shape processing
        shape = getattr(y, "shape", None)
        self._shape = shape if shape is not None else y.shape

        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        # Validate classification targets
        _check_classification_targets(y)
        
        # Process classes
        self.classes_ = []
        self._y = np.empty(y.shape, dtype=int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

        # Validate we have at least 2 classes
        self._validate_n_classes()

        return y

    def _process_regression_targets(self, y):
        """Process regression targets and set shape-related attributes.
        
        REFACTOR: This replicates the EXACT shape processing that was in onedal _fit.
        Original onedal code:
            shape = getattr(y, "shape", None)
            self._shape = shape if shape is not None else y.shape
            # (later, after fit)
            self._y = y if self._shape is None else xp.reshape(y, self._shape)
        
        For now, just store _shape and _y as-is. The reshape happens after onedal fit is complete.
        """
        import sys
        # EXACT replication of original onedal shape processing
        shape = getattr(y, "shape", None)
        self._shape = shape if shape is not None else y.shape
        self._y = y
        print(f"DEBUG _process_regression_targets: _y type={type(self._y)}, _shape={self._shape}", file=sys.stderr)
        return y

    def _fit_validation(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        check_feature_names(self, X, reset=True)
        # Validate n_neighbors parameter
        self._validate_n_neighbors(self.n_neighbors)
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
        # TODO: with verbosity enabled, here it would emit a log saying that it fell
        # back to sklearn, but internally, sklearn will end up calling 'kneighbors'
        # which is overridden in the sklearnex classes, thus it will end up calling
        # oneDAL in the end, but the log will say otherwise. Find a way to make the
        # log consistent with what happens in practice.
        patching_status.and_conditions(
            [
                (
                    not (data[0] is None and method_name in ["predict", "score"]),
                    "Predictions on 'None' data are handled by internal sklearn methods.",
                )
            ]
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
        # requires moving data to host to construct the csr_matrix
        if mode == "connectivity":
            A_ind = self.kneighbors(X, n_neighbors, return_distance=False)
            _, (A_ind,) = _transfer_to_host(A_ind)
            n_queries = A_ind.shape[0]
            A_data = np.ones(n_queries * n_neighbors)

        elif mode == "distance":
            A_data, A_ind = self.kneighbors(X, n_neighbors, return_distance=True)
            _, (A_data, A_ind) = _transfer_to_host(A_data, A_ind)
            A_data = np.reshape(A_data, (-1,))

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                f'or "distance" but got "{mode}" instead'
            )

        n_queries = A_ind.shape[0]
        n_samples_fit = self.n_samples_fit_
        n_nonzero = n_queries * n_neighbors
        A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

        kneighbors_graph = sp.csr_matrix(
            (A_data, np.reshape(A_ind, (-1,)), A_indptr), shape=(n_queries, n_samples_fit)
        )

        return kneighbors_graph

    kneighbors_graph.__doc__ = KNeighborsMixin.kneighbors_graph.__doc__
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

from daal4py.sklearn._utils import is_sparse, sklearn_check_version
from onedal._device_offload import _transfer_to_host
from onedal.utils._array_api import _is_numpy_namespace
from onedal.utils.validation import _check_array, _num_features, _num_samples

from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import get_namespace


class KNeighborsDispatchingBase(oneDALEstimator):
    def _get_weights(self, dist, weights):
        # Adapted from sklearn.neighbors._base._get_weights
        if weights in (None, "uniform"):
            return None
        if weights == "distance":
            # Array API support: get namespace from dist array
            xp, _ = get_namespace(dist)
            # if user attempts to classify a point that was zero distance from one
            # or more training points, those training points are weighted as 1.0
            # and the other points as 0.0
            with (
                xp.errstate(divide="ignore")
                if hasattr(xp, "errstate")
                else np.errstate(divide="ignore")
            ):
                dist = 1.0 / dist
            inf_mask = xp.isinf(dist)
            inf_row = xp.any(inf_mask, axis=1)
            dist[inf_row] = inf_mask[inf_row]
            return dist
        elif callable(weights):
            return weights(dist)
        else:
            raise ValueError(
                "weights not recognized: should be 'uniform', "
                "'distance', or a callable function"
            )

    def _compute_weighted_prediction(self, neigh_dist, neigh_ind, weights_param, y_train):
        """Compute weighted prediction for regression.

        Parameters
        ----------
        neigh_dist : array-like
            Distances to neighbors.
        neigh_ind : array-like
            Indices of neighbors.
        weights_param : str or callable
            Weight parameter ('uniform', 'distance', or callable).
        y_train : array-like
            Training target values.

        Returns
        -------
        array-like
            Predicted values.
        """
        # Array API support: get namespace from input arrays
        xp, _ = get_namespace(neigh_dist, neigh_ind, y_train)

        weights = self._get_weights(neigh_dist, weights_param)

        _y = y_train
        if _y.ndim == 1:
            _y = xp.reshape(_y, (-1, 1))

        if weights is None:
            # Array API: Use take() per row since array API take() only supports 1-D indices
            # Build result by gathering rows one at a time
            gathered_list = []
            for i in range(neigh_ind.shape[0]):
                # Get indices for this sample's neighbors
                sample_indices = neigh_ind[i, ...]  # Shape: (n_neighbors,)
                # Gather those rows from _y
                sample_neighbors = xp.take(
                    _y, sample_indices, axis=0
                )  # Shape: (n_neighbors, n_outputs)
                gathered_list.append(sample_neighbors)
            # Stack and compute mean
            gathered = xp.stack(
                gathered_list, axis=0
            )  # Shape: (n_samples, n_neighbors, n_outputs)
            y_pred = xp.mean(gathered, axis=1)
        else:
            # Create y_pred array - matches original onedal implementation using empty()
            # For Array API arrays (dpctl/dpnp), pass device parameter to match input device
            # For numpy arrays, device parameter is not supported and not needed
            y_pred_shape = (neigh_ind.shape[0], _y.shape[1])
            if not _is_numpy_namespace(xp):
                # Array API: pass device to ensure same device as input
                y_pred = xp.empty(
                    y_pred_shape, dtype=neigh_dist.dtype, device=neigh_ind.device
                )
            else:
                # Numpy: no device parameter
                y_pred = xp.empty(y_pred_shape, dtype=neigh_dist.dtype)
            denom = xp.sum(weights, axis=1)

            for j in range(_y.shape[1]):
                # Array API: Iterate over samples to gather values
                y_col_j = _y[:, j, ...]  # Shape: (n_train_samples,)
                gathered_vals = []
                for i in range(neigh_ind.shape[0]):
                    sample_indices = neigh_ind[i, ...]  # Shape: (n_neighbors,)
                    sample_vals = xp.take(
                        y_col_j, sample_indices, axis=0
                    )  # Shape: (n_neighbors,)
                    gathered_vals.append(sample_vals)
                gathered_j = xp.stack(
                    gathered_vals, axis=0
                )  # Shape: (n_samples, n_neighbors)
                num = xp.sum(gathered_j * weights, axis=1)
                y_pred[:, j, ...] = num / denom

        if y_train.ndim == 1:
            y_pred = xp.reshape(y_pred, (-1,))

        return y_pred

    def _compute_class_probabilities(
        self, neigh_dist, neigh_ind, weights_param, y_train, classes, outputs_2d
    ):
        """Compute class probabilities for classification.

        Parameters
        ----------
        neigh_dist : array-like
            Distances to neighbors.
        neigh_ind : array-like
            Indices of neighbors.
        weights_param : str or callable
            Weight parameter ('uniform', 'distance', or callable).
        y_train : array-like
            Encoded training labels.
        classes : array-like
            Class labels.
        outputs_2d : bool
            Whether output is 2D (multi-output).

        Returns
        -------
        array-like
            Class probabilities.
        """
        from ..utils.validation import _num_samples

        # Transfer all arrays to host to ensure they're on the same queue/device
        # This is needed especially for SPMD where arrays might be on different queues
        _, (neigh_dist, neigh_ind, y_train) = _transfer_to_host(
            neigh_dist, neigh_ind, y_train
        )

        # After transfer, get the array namespace (will be numpy for host arrays)
        xp, _ = get_namespace(neigh_dist, neigh_ind, y_train)

        _y = y_train
        classes_ = classes
        if not outputs_2d:
            _y = xp.reshape(y_train, (-1, 1))
            classes_ = [classes]

        n_queries = neigh_ind.shape[0]

        weights = self._get_weights(neigh_dist, weights_param)
        if weights is None:
            # REFACTOR: Ensure weights is float for array API type promotion
            # neigh_ind is int, so ones_like would give int, but we need float
            # Use neigh_dist dtype to preserve input precision (float32/float64)
            weights = xp.ones_like(neigh_ind, dtype=neigh_dist.dtype)

        probabilities = []
        for k, classes_k in enumerate(classes_):
            # Get predicted labels for each neighbor: shape (n_samples, n_neighbors)
            # _y[:, k] gives training labels for output k, then gather using neigh_ind
            y_col_k = _y[:, k, ...]

            # Array API: Use take() with iteration since take() only supports 1-D indices
            pred_labels_list = []
            for i in range(neigh_ind.shape[0]):
                sample_indices = neigh_ind[i, ...]
                sample_labels = xp.take(y_col_k, sample_indices, axis=0)
                pred_labels_list.append(sample_labels)
            pred_labels = xp.stack(
                pred_labels_list, axis=0
            )  # Shape: (n_queries, n_neighbors)

            proba_k = xp.zeros((n_queries, classes_k.size), dtype=neigh_dist.dtype)

            # Array API: Cannot use fancy indexing __setitem__ like proba_k[all_rows, idx] = ...
            # Instead, build probabilities sample by sample
            proba_list = []
            zero_weight = xp.asarray(0.0, dtype=neigh_dist.dtype)
            for sample_idx in range(n_queries):
                sample_proba = xp.zeros((classes_k.size,), dtype=neigh_dist.dtype)
                # For this sample, accumulate weights for each neighbor's predicted class
                for neighbor_idx in range(pred_labels.shape[1]):
                    class_label = int(pred_labels[sample_idx, neighbor_idx])
                    weight = weights[sample_idx, neighbor_idx]
                    # Update probability for this class using array indexing
                    # Create a mask for this class and add weight where mask is True
                    mask = xp.arange(classes_k.size) == class_label
                    sample_proba = sample_proba + xp.where(mask, weight, zero_weight)
                proba_list.append(sample_proba)
            proba_k = xp.stack(proba_list, axis=0)  # Shape: (n_queries, n_classes)

            # normalize 'votes' into real [0,1] probabilities
            normalizer = xp.sum(proba_k, axis=1)[:, xp.newaxis]
            # Use array scalar for comparison and assignment
            zero_scalar = xp.asarray(0.0, dtype=neigh_dist.dtype)
            one_scalar = xp.asarray(1.0, dtype=neigh_dist.dtype)
            normalizer = xp.where(normalizer == zero_scalar, one_scalar, normalizer)
            proba_k /= normalizer

            probabilities.append(proba_k)

        if not outputs_2d:
            probabilities = probabilities[0]

        return probabilities

    def _validate_n_neighbors(self, n_neighbors):
        if n_neighbors is not None:
            if n_neighbors <= 0:
                raise ValueError("Expected n_neighbors > 0. Got %d" % n_neighbors)
            if not isinstance(n_neighbors, Integral):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" % type(n_neighbors)
                )

    def _set_effective_metric(self):
        """Set effective_metric_ and effective_metric_params_ without validation.

        Used when we need to set metrics but can't call _fit_validation
        (e.g., in SPMD mode with use_raw_input=True where sklearn validation
        would try to convert array API to numpy).
        """
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

        # Convert sklearn metric aliases to canonical names for oneDAL compatibility
        metric_aliases = {
            "cityblock": "manhattan",
            "l1": "manhattan",
            "l2": "euclidean",
        }
        if self.metric in metric_aliases:
            self.effective_metric_ = metric_aliases[self.metric]

        # For minkowski distance, use more efficient methods where available
        if self.metric == "minkowski":
            p = self.effective_metric_params_["p"]
            if p == 1:
                self.effective_metric_ = "manhattan"
            elif p == 2:
                self.effective_metric_ = "euclidean"
            elif p == np.inf:
                self.effective_metric_ = "chebyshev"

    def _validate_n_classes(self):
        """Validate that the classifier has at least 2 classes."""
        length = 0 if self.classes_ is None else len(self.classes_)
        if length < 2:
            raise ValueError(
                f"The number of classes has to be greater than one; got {length}"
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
        - n_neighbors is within valid bounds if provided

        Note: Feature validation (count, names, etc.) happens in validate_data
        called by _onedal_kneighbors, so we don't duplicate it here.
        """
        # Validate n_neighbors bounds if provided
        if n_neighbors is not None:
            # Determine if query is the training set
            query_is_train = X is None or (hasattr(self, "_fit_X") and X is self._fit_X)
            self._validate_kneighbors_bounds(
                n_neighbors, query_is_train, X if X is not None else self._fit_X
            )

    def _fit_validation(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        # check_feature_names(self, X, reset=True)
        # Validate n_neighbors parameter
        self._validate_n_neighbors(self.n_neighbors)

        # Set effective metric and parameters
        self._set_effective_metric()

        if not isinstance(X, (KDTree, BallTree, _sklearn_NeighborsBase)):
            # Use _check_array like main branch, but with array API dtype support
            # Get array namespace for array API support
            # Don't check for NaN - let oneDAL handle it (will fallback to sklearn if needed)
            xp, _ = get_namespace(X)
            self._fit_X = _check_array(
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse=True,
                force_all_finite=False,
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

        # Only delete _onedal_estimator if it's an instance attribute, not a class attribute
        # (SPMD classes define _onedal_estimator as a staticmethod at class level)
        if "_onedal_estimator" in self.__dict__:
            delattr(self, "_onedal_estimator")
        # To cover test case when we pass patched
        # estimator as an input for other estimator
        if isinstance(X, _sklearn_NeighborsBase):
            self._fit_X = X._fit_X
            self._tree = X._tree
            self._fit_method = X._fit_method
            self.n_samples_fit_ = X.n_samples_fit_
            self.n_features_in_ = X.n_features_in_
            # Check if X has _onedal_estimator as an instance attribute (not class attribute)
            if "_onedal_estimator" in X.__dict__:
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
            not is_sparse(data[0]), "Sparse input is not supported."
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
                # Array API support: get namespace from y
                y_input = data[1]
                xp, _ = get_namespace(y_input)
                y = xp.asarray(y_input)
                if is_classifier:
                    # Use numpy for unique (standard sklearn pattern)
                    class_count = len(np.unique(np.asarray(y)))
            # Only access _onedal_estimator if it's an instance attribute (not a class-level staticmethod)
            if "_onedal_estimator" in self.__dict__:
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
            # Check if _onedal_estimator is an instance attribute (model was trained)
            # For SPMD classes, _onedal_estimator is a class-level staticmethod, so we check __dict__
            patching_status.and_condition(
                "_onedal_estimator" in self.__dict__, "oneDAL model was not trained."
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

        # construct CSR matrix representation of the k-NN graph
        # requires moving data to host to construct the csr_matrix
        if mode == "connectivity":
            A_ind = self._onedal_estimator.kneighbors(
                X, n_neighbors, return_distance=False
            )
            # Transfer to host - after this, arrays are numpy
            _, (A_ind,) = _transfer_to_host(A_ind)
            n_queries = A_ind.shape[0]
            # Use numpy after transfer to host
            A_data = np.ones(n_queries * n_neighbors)

        elif mode == "distance":
            A_data, A_ind = self._onedal_estimator.kneighbors(
                X, n_neighbors, return_distance=True
            )
            # Transfer to host - after this, arrays are numpy
            _, (A_data, A_ind) = _transfer_to_host(A_data, A_ind)
            # Use numpy after transfer to host
            A_data = np.reshape(A_data, (-1,))

        else:
            raise ValueError(
                'Unsupported mode, must be one of "connectivity", '
                f'or "distance" but got "{mode}" instead'
            )

        n_queries = A_ind.shape[0]
        n_samples_fit = self.n_samples_fit_
        n_nonzero = n_queries * n_neighbors
        # Use numpy after transfer to host
        A_indptr = np.arange(0, n_nonzero + 1, n_neighbors)

        kneighbors_graph = sp.csr_matrix(
            (A_data, np.reshape(A_ind, (-1,)), A_indptr), shape=(n_queries, n_samples_fit)
        )

        return kneighbors_graph

    kneighbors_graph.__doc__ = KNeighborsMixin.kneighbors_graph.__doc__

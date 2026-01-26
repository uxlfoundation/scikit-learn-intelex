# ===============================================================================
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
# ===============================================================================

from sklearn.metrics import accuracy_score
from sklearn.neighbors._classification import (
    KNeighborsClassifier as _sklearn_KNeighborsClassifier,
)
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal.neighbors import KNeighborsClassifier as onedal_KNeighborsClassifier

from .._config import get_config
from .._device_offload import dispatch, wrap_output_data
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import check_feature_names, validate_data
from .common import KNeighborsDispatchingBase


@enable_array_api
@control_n_jobs(
    decorated_methods=["fit", "predict", "predict_proba", "kneighbors", "score"]
)
class KNeighborsClassifier(KNeighborsDispatchingBase, _sklearn_KNeighborsClassifier):
    __doc__ = _sklearn_KNeighborsClassifier.__doc__
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_KNeighborsClassifier._parameter_constraints
        }

    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )

    def fit(self, X, y):
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_KNeighborsClassifier.fit,
            },
            X,
            y,
        )
        return self

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)

        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_KNeighborsClassifier.predict,
            },
            X,
        )

    @wrap_output_data
    def predict_proba(self, X):
        check_is_fitted(self)

        return dispatch(
            self,
            "predict_proba",
            {
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": _sklearn_KNeighborsClassifier.predict_proba,
            },
            X,
        )

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)

        return dispatch(
            self,
            "score",
            {
                "onedal": self.__class__._onedal_score,
                "sklearn": _sklearn_KNeighborsClassifier.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        # Validate n_neighbors parameter first
        if n_neighbors is not None:
            self._validate_n_neighbors(n_neighbors)

        check_is_fitted(self)

        # Validate kneighbors parameters (inherited from KNeighborsDispatchingBase)
        self._kneighbors_validation(X, n_neighbors)

        return dispatch(
            self,
            "kneighbors",
            {
                "onedal": self.__class__._onedal_kneighbors,
                "sklearn": _sklearn_KNeighborsClassifier.kneighbors,
            },
            X,
            n_neighbors=n_neighbors,
            return_distance=return_distance,
        )

    def _onedal_fit(self, X, y, queue=None):
        xp, _ = get_namespace(X)

        # Validation step
        if not get_config()["use_raw_input"]:
            X, y = validate_data(
                self,
                X,
                y,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
            )

        # SPMD mode: skip validation but still set effective metric
        self._set_effective_metric()

        # Process classification targets before passing to onedal
        self._process_classification_targets(
            y, skip_validation=get_config()["use_raw_input"]
        )

        # Call onedal backend
        onedal_params = {
            "n_neighbors": self.n_neighbors,
            "weights": self.weights,
            "algorithm": self.algorithm,
            "metric": self.effective_metric_,
            "p": self.effective_metric_params_["p"],
        }

        self._onedal_estimator = onedal_KNeighborsClassifier(**onedal_params)
        self._onedal_estimator.requires_y = get_requires_y_tag(self)
        self._onedal_estimator.effective_metric_ = self.effective_metric_
        self._onedal_estimator.effective_metric_params_ = self.effective_metric_params_
        self._onedal_estimator.classes_ = self.classes_
        self._onedal_estimator._y = self._y
        self._onedal_estimator.outputs_2d_ = self.outputs_2d_
        self._onedal_estimator._shape = self._shape

        self._onedal_estimator.fit(X, y, queue=queue)

        # Post-processing
        self._save_attributes()

    def _compute_class_probabilities(
        self, neigh_dist, neigh_ind, weights_param, y_train, classes, outputs_2d
    ):
        """Compute class probabilities for classification.

        Parameters
        ----------
        neigh_dist : array
            Distances to neighbors.
        neigh_ind : array
            Indices of neighbors.
        weights_param : {'uniform', 'distance'}, callable or None
            Weight parameter.
        y_train : array
            Encoded training labels.
        classes : array or list of arrays
            Class labels.
        outputs_2d : bool
            Whether output is 2D (multi-output).

        Returns
        -------
        probabilities : array or list of arrays
            Class probabilities.
        """
        # Array API support: get namespace from input arrays
        xp, _ = get_namespace(neigh_dist, neigh_ind, y_train)

        _y = y_train
        classes_ = classes
        if not outputs_2d:
            _y = xp.reshape(y_train, (-1, 1))
            classes_ = [classes]

        n_queries = neigh_ind.shape[0]

        weights = self._get_weights(neigh_dist, weights_param)
        if weights is None:
            # Ensure weights is float for array API type promotion
            # neigh_ind is int, so ones_like would give int, but we need float
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

    def _predict_skl_classification(self, X):
        """SKL prediction path for classification - calls kneighbors, computes predictions.

        This method handles X=None (LOOCV) properly by calling self.kneighbors which
        has the query_is_train logic.

        Parameters
        ----------
        X : array-like or None
            Query samples, or None for LOOCV.

        Returns
        -------
        y_pred : array
            Predicted class labels.
        """
        neigh_dist, neigh_ind = self.kneighbors(X)
        proba = self._compute_class_probabilities(
            neigh_dist, neigh_ind, self.weights, self._y, self.classes_, self.outputs_2d_
        )
        # Array API support: get namespace from probability array
        xp, _ = get_namespace(proba)

        if not self.outputs_2d_:
            # Single output: classes_[argmax(proba, axis=1)]
            return self.classes_[xp.argmax(proba, axis=1)]
        else:
            # Multi-output: apply argmax separately for each output
            result = [
                classes_k[xp.argmax(proba_k, axis=1)]
                for classes_k, proba_k in zip(self.classes_, proba.T)
            ]
            return xp.asarray(result).T

    def _onedal_predict(self, X, queue=None):
        # Use the helper method (calls kneighbors + computes prediction)
        # This properly handles X=None (LOOCV) case
        # Note: X validation happens in kneighbors
        return self._predict_skl_classification(X)

    def _onedal_predict_proba(self, X, queue=None):
        # Call kneighbors through sklearnex (self.kneighbors is the sklearnex method)
        # This properly handles X=None case (LOOCV) with query_is_train logic
        # Note: X validation happens in kneighbors
        neigh_dist, neigh_ind = self.kneighbors(X)

        # Use the helper method to compute class probabilities
        return self._compute_class_probabilities(
            neigh_dist, neigh_ind, self.weights, self._y, self.classes_, self.outputs_2d_
        )

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        # Only skip validation when use_raw_input=True (SPMD mode)
        use_raw_input = get_config()["use_raw_input"]

        if X is not None and not use_raw_input:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                accept_sparse="csr",
                reset=False,
            )

        # onedal backend now handles all logic:
        # - X=None case (query_is_train)
        # - kd_tree sorting
        # - removing self from results
        return self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        # Get predictions
        # y_pred = self._onedal_predict(X, queue=queue)

        # Convert array API to numpy for sklearn's accuracy_score using _transfer_to_host
        # This properly handles Array API arrays that don't allow implicit conversion
        # _, (y, y_pred, sample_weight) = _transfer_to_host(y, y_pred, sample_weight)

        return accuracy_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    def _save_attributes(self):
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        self._y = self._onedal_estimator._y
        self._fit_method = self._onedal_estimator._fit_method
        self.outputs_2d_ = self._onedal_estimator.outputs_2d_
        self._tree = self._onedal_estimator._tree

    fit.__doc__ = _sklearn_KNeighborsClassifier.fit.__doc__
    predict.__doc__ = _sklearn_KNeighborsClassifier.predict.__doc__
    predict_proba.__doc__ = _sklearn_KNeighborsClassifier.predict_proba.__doc__
    score.__doc__ = _sklearn_KNeighborsClassifier.score.__doc__
    kneighbors.__doc__ = _sklearn_KNeighborsClassifier.kneighbors.__doc__

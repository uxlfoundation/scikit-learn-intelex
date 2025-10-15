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

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors._classification import (
    KNeighborsClassifier as _sklearn_KNeighborsClassifier,
)
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal.neighbors import KNeighborsClassifier as onedal_KNeighborsClassifier

from .._device_offload import dispatch, wrap_output_data
from ..utils.validation import check_feature_names, validate_data
from .common import KNeighborsDispatchingBase


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
        import sys
        print(f"DEBUG KNeighborsClassifier.fit START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}, y type={type(y)}", file=sys.stderr)
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
        print(f"DEBUG KNeighborsClassifier.fit END: _fit_X type={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
        return self

    @wrap_output_data
    def predict(self, X):
        import sys
        print(f"DEBUG KNeighborsClassifier.predict START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        check_is_fitted(self)
        check_feature_names(self, X, reset=False)
        result = dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_KNeighborsClassifier.predict,
            },
            X,
        )
        print(f"DEBUG KNeighborsClassifier.predict END: result type={type(result)}", file=sys.stderr)
        return result

    @wrap_output_data
    def predict_proba(self, X):
        import sys
        print(f"DEBUG KNeighborsClassifier.predict_proba START: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
        check_is_fitted(self)
        check_feature_names(self, X, reset=False)
        result = dispatch(
            self,
            "predict_proba",
            {
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": _sklearn_KNeighborsClassifier.predict_proba,
            },
            X,
        )
        print(f"DEBUG KNeighborsClassifier.predict_proba END: result type={type(result)}", file=sys.stderr)
        return result

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        import sys
        print(f"DEBUG KNeighborsClassifier.score START: X type={type(X)}, y type={type(y)}", file=sys.stderr)
        check_is_fitted(self)
        check_feature_names(self, X, reset=False)
        result = dispatch(
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
        print(f"DEBUG KNeighborsClassifier.score END: result={result}", file=sys.stderr)
        return result

    @wrap_output_data
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True):
        import sys
        print(f"DEBUG KNeighborsClassifier.kneighbors START: X type={type(X)}, n_neighbors={n_neighbors}, return_distance={return_distance}", file=sys.stderr)
        check_is_fitted(self)
        if X is not None:
            check_feature_names(self, X, reset=False)
        
        # Validate kneighbors parameters (inherited from KNeighborsDispatchingBase)
        self._kneighbors_validation(X, n_neighbors)
        
        result = dispatch(
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
        print(f"DEBUG KNeighborsClassifier.kneighbors END: result type={type(result)}", file=sys.stderr)
        return result

    def _onedal_fit(self, X, y, queue=None):
        import sys
        print(f"DEBUG KNeighborsClassifier._onedal_fit START: X type={type(X)}, y type={type(y)}", file=sys.stderr)
        
        # REFACTOR: Use validate_data from sklearnex.utils.validation to convert pandas to numpy
        X, y = validate_data(
            self, X, y, dtype=[np.float64, np.float32], accept_sparse="csr"
        )
        print(f"DEBUG: After validate_data, X type={type(X)}, y type={type(y)}", file=sys.stderr)
        
        # REFACTOR STEP 1: Process classification targets in sklearnex before passing to onedal
        print(f"DEBUG: Processing classification targets in sklearnex", file=sys.stderr)
        y_processed = self._process_classification_targets(y)
        print(f"DEBUG: After _process_classification_targets, y_processed type={type(y_processed)}", file=sys.stderr)
        
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
        
        # REFACTOR: Pass both original and processed targets to onedal
        # onedal needs the processed classes_ and _y attributes that we just set
        self._onedal_estimator.classes_ = self.classes_
        self._onedal_estimator._y = self._y
        self._onedal_estimator.outputs_2d_ = self.outputs_2d_
        print(f"DEBUG: Set onedal_estimator.classes_={self._onedal_estimator.classes_}", file=sys.stderr)
        print(f"DEBUG: Set onedal_estimator._y shape={self._onedal_estimator._y.shape}", file=sys.stderr)
        
        print(f"DEBUG KNeighborsClassifier._onedal_fit: Calling onedal_estimator.fit with X and original y", file=sys.stderr)
        # Pass original y to onedal - it will use the pre-set classes_ and _y attributes we just assigned
        self._onedal_estimator.fit(X, y, queue=queue)
        print(f"DEBUG KNeighborsClassifier._onedal_fit: After fit, calling _save_attributes", file=sys.stderr)

        self._save_attributes()
        print(f"DEBUG KNeighborsClassifier._onedal_fit END: self._fit_X type={type(getattr(self, '_fit_X', 'NOT_SET'))}", file=sys.stderr)

    def _onedal_predict(self, X, queue=None):
        import sys
        print(f"DEBUG KNeighborsClassifier._onedal_predict START: X type={type(X)}", file=sys.stderr)
        result = self._onedal_estimator.predict(X, queue=queue)
        print(f"DEBUG KNeighborsClassifier._onedal_predict END: result type={type(result)}", file=sys.stderr)
        return result

    def _onedal_predict_proba(self, X, queue=None):
        import sys
        print(f"DEBUG KNeighborsClassifier._onedal_predict_proba START: X type={type(X)}", file=sys.stderr)
        result = self._onedal_estimator.predict_proba(X, queue=queue)
        print(f"DEBUG KNeighborsClassifier._onedal_predict_proba END: result type={type(result)}", file=sys.stderr)
        return result

    def _onedal_kneighbors(
        self, X=None, n_neighbors=None, return_distance=True, queue=None
    ):
        import sys
        print(f"DEBUG KNeighborsClassifier._onedal_kneighbors START: X type={type(X)}, n_neighbors={n_neighbors}, return_distance={return_distance}", file=sys.stderr)
        result = self._onedal_estimator.kneighbors(
            X, n_neighbors, return_distance, queue=queue
        )
        print(f"DEBUG KNeighborsClassifier._onedal_kneighbors END: result type={type(result)}", file=sys.stderr)
        return result

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        import sys
        print(f"DEBUG KNeighborsClassifier._onedal_score START: X type={type(X)}, y type={type(y)}", file=sys.stderr)
        result = accuracy_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )
        print(f"DEBUG KNeighborsClassifier._onedal_score END: result={result}", file=sys.stderr)
        return result

    def _save_attributes(self):
        import sys
        print(f"DEBUG KNeighborsClassifier._save_attributes START", file=sys.stderr)
        self.classes_ = self._onedal_estimator.classes_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.n_samples_fit_ = self._onedal_estimator.n_samples_fit_
        self._fit_X = self._onedal_estimator._fit_X
        print(f"DEBUG KNeighborsClassifier._save_attributes: _fit_X type={type(self._fit_X)}", file=sys.stderr)
        self._y = self._onedal_estimator._y
        print(f"DEBUG KNeighborsClassifier._save_attributes: _y type={type(self._y)}", file=sys.stderr)
        self._fit_method = self._onedal_estimator._fit_method
        self.outputs_2d_ = self._onedal_estimator.outputs_2d_
        self._tree = self._onedal_estimator._tree
        print(f"DEBUG KNeighborsClassifier._save_attributes END", file=sys.stderr)

    fit.__doc__ = _sklearn_KNeighborsClassifier.fit.__doc__
    predict.__doc__ = _sklearn_KNeighborsClassifier.predict.__doc__
    predict_proba.__doc__ = _sklearn_KNeighborsClassifier.predict_proba.__doc__
    score.__doc__ = _sklearn_KNeighborsClassifier.score.__doc__
    kneighbors.__doc__ = _sklearn_KNeighborsClassifier.kneighbors.__doc__
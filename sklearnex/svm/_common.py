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

import warnings
from numbers import Number, Real

import numpy as np
from scipy import sparse as sp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.utils.validation import get_requires_y_tag
from onedal.utils.validation import _check_array, _check_X_y, _column_or_1d

from .._config import config_context, get_config
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils.validation import validate_data


class BaseSVM(oneDALEstimator):

    @property
    def _dual_coef_(self):
        return self._dualcoef_

    @_dual_coef_.setter
    def _dual_coef_(self, value):
        self._dualcoef_ = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.dual_coef_ = value
            if hasattr(self._onedal_estimator, "_onedal_model"):
                del self._onedal_estimator._onedal_model

    @_dual_coef_.deleter
    def _dual_coef_(self):
        del self._dualcoef_

    @property
    def intercept_(self):
        return self._icept_

    @intercept_.setter
    def intercept_(self, value):
        self._icept_ = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator.intercept_ = value
            if hasattr(self._onedal_estimator, "_onedal_model"):
                del self._onedal_estimator._onedal_model

    @intercept_.deleter
    def intercept_(self):
        del self._icept_

    def _onedal_gpu_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(f"sklearn.{method_name}")
        patching_status.and_conditions([(False, "GPU offloading is not supported.")])
        return patching_status

    def _onedal_cpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.svm.{class_name}.{method_name}"
        )
        if method_name == "fit":
            patching_status.and_conditions(
                [
                    (
                        self.kernel in ["linear", "rbf", "poly", "sigmoid"],
                        f'Kernel is "{self.kernel}" while '
                        '"linear", "rbf", "poly" and "sigmoid" are only supported.',
                    )
                ]
            )
            return patching_status
        inference_methods = (
            ["predict", "score"]
            if class_name.endswith("R")
            else ["predict", "predict_proba", "decision_function", "score"]
        )
        if method_name in inference_methods:
            patching_status.and_conditions(
                [(hasattr(self, "_onedal_estimator"), "oneDAL model was not trained.")]
            )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {class_name}")

    def _compute_gamma_sigma(self, X):
        # only run extended conversion if kernel is not linear
        # set to a value = 1.0, so gamma will always be passed to
        # the onedal estimator as a float type
        if self.kernel == "linear":
            return 1.0

        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                if sp.issparse(X):
                    # var = E[X^2] - E[X]^2
                    X_sc = (X.multiply(X)).mean() - (X.mean()) ** 2
                else:
                    X_sc = X.var()
                _gamma = 1.0 / (X.shape[1] * X_sc) if X_sc != 0 else 1.0
            elif self.gamma == "auto":
                _gamma = 1.0 / X.shape[1]
            else:
                raise ValueError(
                    "When 'gamma' is a string, it should be either 'scale' or "
                    "'auto'. Got '{}' instead.".format(self.gamma)
                )
        else:
            if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
                if isinstance(self.gamma, Real):
                    if self.gamma <= 0:
                        msg = (
                            f"gamma value must be > 0; {self.gamma!r} is invalid. Use"
                            " a positive number or use 'auto' to set gamma to a"
                            " value of 1 / n_features."
                        )
                        raise ValueError(msg)
                    _gamma = self.gamma
                else:
                    msg = (
                        "The gamma value should be set to 'scale', 'auto' or a"
                        f" positive float value. {self.gamma!r} is not a valid option"
                    )
                    raise ValueError(msg)
            else:
                _gamma = self.gamma
        return _gamma

    def _onedal_fit_checks(self, X, y, sample_weight=None):
        if hasattr(self, "decision_function_shape"):
            if self.decision_function_shape not in ("ovr", "ovo", None):
                raise ValueError(
                    f"decision_function_shape must be either 'ovr' or 'ovo', "
                    f"got {self.decision_function_shape}."
                )

        if y is None:
            if get_requires_y_tag(self):
                raise ValueError(
                    f"This {self.__class__.__name__} estimator "
                    f"requires y to be passed, but the target y is None."
                )
        # finite check occurs in onedal estimator
        X, y = validate_data(
            self,
            X,
            y,
            dtype=[np.float64, np.float32],
            ensure_all_finite=False,
            accept_sparse="csr",
        )
        y = self._validate_targets(y)
        sample_weight = self._get_sample_weight(X, y, sample_weight)
        return X, y, sample_weight

    def _get_sample_weight(self, X, y, sample_weight):
        n_samples = X.shape[0]
        dtype = X.dtype
        if n_samples == 1:
            raise ValueError("n_samples=1")

        sample_weight = np.ascontiguousarray(
            [] if sample_weight is None else sample_weight, dtype=np.float64
        )

        sample_weight_count = sample_weight.shape[0]
        if sample_weight_count != 0 and sample_weight_count != n_samples:
            raise ValueError(
                "sample_weight and X have incompatible shapes: "
                "%r vs %r\n"
                "Note: Sparse matrices cannot be indexed w/"
                "boolean masks (use `indices=True` in CV)."
                % (len(sample_weight), X.shape)
            )

        if sample_weight_count == 0:
            if not isinstance(self, ClassifierMixin) or self.class_weight_ is None:
                return None
            sample_weight = np.ones(n_samples, dtype=dtype)
        elif isinstance(sample_weight, Number):
            sample_weight = np.full(n_samples, sample_weight, dtype=dtype)
        else:
            sample_weight = _check_array(
                sample_weight,
                accept_sparse=False,
                ensure_2d=False,
                dtype=dtype,
                order="C",
            )
            if sample_weight.ndim != 1:
                raise ValueError("Sample weights must be 1D array or scalar")

            if sample_weight.shape != (n_samples,):
                raise ValueError(
                    "sample_weight.shape == {}, expected {}!".format(
                        sample_weight.shape, (n_samples,)
                    )
                )

        if np.all(sample_weight <= 0):
            if "nusvc" in self.__module__:
                raise ValueError("negative dimensions are not allowed")
            else:
                raise ValueError(
                    "Invalid input - all samples have zero or negative weights."
                )

        return sample_weight


class BaseSVC(BaseSVM):
    def _compute_balanced_class_weight(self, y):
        y_ = _column_or_1d(y)
        classes, _ = np.unique(y_, return_inverse=True)

        le = LabelEncoder()
        y_ind = le.fit_transform(y_)
        if not np.isin(classes, le.classes_).all():
            raise ValueError("classes should have valid labels that are in y")

        recip_freq = len(y_) / (len(le.classes_) * np.bincount(y_ind).astype(np.float64))
        return recip_freq[le.transform(classes)]

    def _fit_proba(self, X, y, sample_weight=None, queue=None):
        # TODO: rewrite this method when probabilities output is implemented in oneDAL

        # LibSVM uses the random seed to control cross-validation for probability generation
        # CalibratedClassifierCV with "prefit" does not use an RNG nor a seed. This may
        # impact users without their knowledge, so display a warning.
        if self.random_state is not None:
            warnings.warn(
                "random_state does not influence oneDAL SVM results",
                RuntimeWarning,
            )

        params = self.get_params()
        params["probability"] = False
        params["decision_function_shape"] = "ovr"
        clf_base = self.__class__(**params)

        # We use stock metaestimators below, so the only way
        # to pass a queue is using config_context.
        cfg = get_config()
        cfg["target_offload"] = queue
        with config_context(**cfg):
            clf_base.fit(X, y)
            self.clf_prob = CalibratedClassifierCV(
                clf_base,
                ensemble=False,
                cv="prefit",
                method="sigmoid",
            ).fit(X, y)

    def _save_attributes(self):
        self.support_vectors_ = self._onedal_estimator.support_vectors_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.fit_status_ = 0
        self.dual_coef_ = self._onedal_estimator.dual_coef_
        self.shape_fit_ = self._onedal_estimator.class_weight_
        self.classes_ = self._onedal_estimator.classes_
        if isinstance(self, ClassifierMixin) or not sklearn_check_version("1.2"):
            self.class_weight_ = self._onedal_estimator.class_weight_
        self.support_ = self._onedal_estimator.support_

        self._icept_ = self._onedal_estimator.intercept_
        self._n_support = self._onedal_estimator._n_support
        self._sparse = False
        self._gamma = self._onedal_estimator._gamma
        if self.probability:
            length = int(len(self.classes_) * (len(self.classes_) - 1) / 2)
            self._probA = np.zeros(length)
            self._probB = np.zeros(length)
        else:
            self._probA = np.empty(0)
            self._probB = np.empty(0)

        self._dualcoef_ = self.dual_coef_

        if sklearn_check_version("1.1"):
            length = int(len(self.classes_) * (len(self.classes_) - 1) / 2)
            self.n_iter_ = np.full((length,), self._onedal_estimator.n_iter_)


class BaseSVR(BaseSVM):
    def _save_attributes(self):
        self.support_vectors_ = self._onedal_estimator.support_vectors_
        self.n_features_in_ = self._onedal_estimator.n_features_in_
        self.fit_status_ = 0
        self.dual_coef_ = self._onedal_estimator.dual_coef_
        self.shape_fit_ = self._onedal_estimator.shape_fit_
        self.support_ = self._onedal_estimator.support_

        self._icept_ = self._onedal_estimator.intercept_
        self._n_support = [self.support_vectors_.shape[0]]
        self._sparse = False
        self._gamma = self._onedal_estimator._gamma
        self._probA = None
        self._probB = None

        if sklearn_check_version("1.1"):
            self.n_iter_ = self._onedal_estimator.n_iter_

        self._dualcoef_ = self.dual_coef_

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return r2_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

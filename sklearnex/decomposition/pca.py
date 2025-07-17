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

import logging

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2024, "P", 100)):
    from math import sqrt
    from numbers import Integral
    from warnings import warn

    import numpy as np
    from scipy.sparse import issparse
    from sklearn.decomposition._pca import _infer_dimension
    from sklearn.utils.extmath import stable_cumsum
    from sklearn.utils.validation import check_is_fitted

    from daal4py.sklearn._n_jobs_support import control_n_jobs
    from daal4py.sklearn._utils import sklearn_check_version

    from .._device_offload import dispatch, wrap_output_data
    from .._utils import PatchingConditionsChain, register_hyperparameters
    from ..base import oneDALEstimator
    from ..utils._array_api import get_namespace
    from ..utils.validation import validate_data

    if sklearn_check_version("1.1") and not sklearn_check_version("1.2"):
        from sklearn.utils import check_scalar

    if sklearn_check_version("1.2"):
        from sklearn.utils._param_validation import StrOptions

    from sklearn.decomposition import PCA as _sklearn_PCA

    from onedal.common.hyperparameters import get_hyperparameters
    from onedal.decomposition import PCA as onedal_PCA
    from onedal.utils._array_api import _is_numpy_namespace
    from onedal.utils.validation import _num_features, _num_samples

    @register_hyperparameters({"fit": get_hyperparameters("pca", "train")})
    @control_n_jobs(decorated_methods=["fit", "transform", "fit_transform"])
    class PCA(oneDALEstimator, _sklearn_PCA):
        __doc__ = _sklearn_PCA.__doc__

        if sklearn_check_version("1.2"):
            _parameter_constraints: dict = {**_sklearn_PCA._parameter_constraints}
            # "onedal_svd" solver uses oneDAL's PCA-SVD algorithm
            # and required for testing purposes to fully enable it in future.
            # "covariance_eigh" solver is added for ability to explicitly request
            # oneDAL's PCA-Covariance algorithm using any sklearn version < 1.5.
            _parameter_constraints["svd_solver"] = [
                StrOptions(
                    _parameter_constraints["svd_solver"][0].options
                    | {"onedal_svd", "covariance_eigh"}
                )
            ]

        if sklearn_check_version("1.1"):

            def __init__(
                self,
                n_components=None,
                *,
                copy=True,
                whiten=False,
                svd_solver="auto",
                tol=0.0,
                iterated_power="auto",
                n_oversamples=10,
                power_iteration_normalizer="auto",
                random_state=None,
            ):
                self.n_components = n_components
                self.copy = copy
                self.whiten = whiten
                self.svd_solver = svd_solver
                self.tol = tol
                self.iterated_power = iterated_power
                self.n_oversamples = n_oversamples
                self.power_iteration_normalizer = power_iteration_normalizer
                self.random_state = random_state

        else:

            def __init__(
                self,
                n_components=None,
                copy=True,
                whiten=False,
                svd_solver="auto",
                tol=0.0,
                iterated_power="auto",
                random_state=None,
            ):
                self.n_components = n_components
                self.copy = copy
                self.whiten = whiten
                self.svd_solver = svd_solver
                self.tol = tol
                self.iterated_power = iterated_power
                self.random_state = random_state

        _onedal_PCA = staticmethod(onedal_PCA)

        def _onedal_supported(self, method_name, *data):
            class_name = self.__class__.__name__
            patching_status = PatchingConditionsChain(
                f"sklearn.decomposition.{class_name}.{method_name}"
            )
            X = data[0]

            if method_name in ["fit", "fit_transform"]:
                # pulling shape of the input is required before offloading
                # due to the nature of sklearn's PCA._fit routine, which is
                # behind a ``validate_data`` call and cannot be used
                # without a performance impact
                n_samples = num_samples(X)
                n_features = num_features(X)
                # in the case that the code falls back to sklearn
                # self._fit_svd_solver will be clobbered in ``PCA._fit``
                # setting values in dispatching is generally forbidden, but
                # must be done in this case due to the sklearn estimator
                # design.
                self._fit_svd_solver = (
                    self._select_svd_solver(n_samples, n_features)
                    if self.svd_solver == "auto"
                    else self.svd_solver
                )
                # Use oneDAL in the following cases:
                # 1. "onedal_svd" solver is explicitly set
                # 2. solver is set to "covariance_eigh"
                # 3. solver is set to "full" and sklearn version < 1.5
                # 4. solver is set to "auto" and dispatched to "full"
                force_solver = self._fit_svd_solver == "full" and (
                    not sklearn_check_version("1.5") or self.svd_solver == "auto"
                )

                patching_status.and_conditions(
                    [
                        (
                            n_features < 2 * n_samples,
                            "Data shape is not compatible.",
                        ),
                        (
                            force_solver
                            or self._fit_svd_solver in ["covariance_eigh", "onedal_svd"],
                            (
                                "Only 'covariance_eigh' and 'onedal_svd' "
                                "solvers are supported."
                                if sklearn_check_version("1.5")
                                else "Only 'full', 'covariance_eigh' and 'onedal_svd' "
                                "solvers are supported."
                            ),
                        ),
                        (not issparse(X), "oneDAL PCA does not support sparse data"),
                    ]
                )
                return patching_status

            if method_name == "transform":
                patching_status.and_conditions(
                    [
                        (
                            hasattr(self, "_onedal_estimator"),
                            "oneDAL model was not trained",
                        ),
                    ]
                )
                return patching_status

            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        _onedal_cpu_supported = _onedal_supported
        _onedal_gpu_supported = _onedal_supported

        def _validate_n_components(self, n_components, n_samples, n_features):
            # This reproduces the initial n_components validation in PCA._fit_full
            # Also a maintenance burden, but is isolated for compartmentalization
            if n_components == "mle":
                if n_samples < n_features:
                    raise ValueError(
                        "n_components='mle' is only supported if n_samples >= n_features"
                    )
            elif not 0 <= n_components <= min(n_samples, n_features):
                raise ValueError(
                    "n_components=%r must be between 0 and "
                    "min(n_samples, n_features)=%r with "
                    "svd_solver='full'" % (n_components, min(n_samples, n_features))
                )
            elif not sklearn_check_version("1.2") and n_components >= 1:
                if not isinstance(n_components, Integral):
                    raise ValueError(
                        "n_components=%r must be of type int "
                        "when greater than or equal to 1, "
                        "was of type=%r" % (n_components, type(n_components))
                    )

        def _postprocess_n_components(self):
            # this method extracts aspects of post-processing located in
            # PCA._fit_full which cannot be re-used.  It is isolated for
            if self.n_components == "mle":
                return _infer_dimension(
                    self._onedal_estimator.explained_variance_, self.n_samples_
                )
            else:
                ratio_cumsum = stable_cumsum(
                    self._onedal_estimator.explained_variance_ratio_
                )
                return np.searchsorted(ratio_cumsum, self.n_components, side="right") + 1

        def _compute_noise_variance(self, n_sf_min, xp=np):
            # This varies from sklearn, but not sure why (and is undocumented from the
            # original implementation in sklearnex)
            if self.n_components_ < n_sf_min:
                if len(self.explained_variance_) == n_sf_min:
                    return xp.mean(self.explained_variance_)
                elif len(self.explained_variance_) < n_sf_min:
                    resid_var = xp.sum(self._onedal_estimator.var_) - xp.sum(
                        self.explained_variance_
                    )
                    return resid_var / (n_sf_min - n_components)
            else:
                return 0.0

        if sklearn_check_version("1.1"):

            def _select_svd_solver(self, n_samples, n_features):
                n_sf_min = min(n_samples, n_features)
                n_components = (
                    n_sf_min if self.n_components is None else self.n_components
                )
                # This is matching aspects of sklearn.decomposition.PCA's ``_fit`` method
                # Must be done this way as the logic hidden behind a ``validate_data`` call
                # in sklearn cannot be reused without performance loss. This is likely to be
                # high maintenance, but is written to be as simple and straightforward as
                # possible.
                if (
                    sklearn_check_version("1.5")
                    and n_features <= 1_000
                    and n_samples >= 10 * n_features
                ):
                    return "covariance_eigh"
                elif max(n_samples, n_features) <= 500 or n_components == "mle":
                    return "full"
                elif 1 <= n_components < 0.8 * n_sf_min:
                    return "randomized"
                else:
                    return "full"

        else:

            def _select_svd_solver(self, n_samples, n_features):
                n_sf_min = min(n_samples, n_features)
                n_components = (
                    n_sf_min if self.n_components is None else self.n_components
                )

                if n_components == "mle":
                    return "full"
                else:
                    # check if sklearnex is faster than randomized sklearn
                    # Refer to daal4py, this is legacy and should be either
                    # regenerated or removed. Refactored from daal4py to
                    # remove unnecessary math.
                    d4p_analysis = (
                        n_features
                        * (9.779873e-11 * n_components - 1.122062e-11 * n_features)
                        + 1.127905e-09 * n_samples
                    )
                    if n_components >= 1 and d4p_analysis <= 0:
                        return "randomized"
                    else:
                        return "full"

        def fit(self, X):
            if sklearn_check_version("1.2"):
                self._validate_params()
            elif sklearn_check_version("1.1"):
                check_scalar(
                    self.n_oversamples,
                    "n_oversamples",
                    min_val=1,
                    target_type=Integral,
                )

            dispatch(
                self,
                "fit",
                {
                    "onedal": self.__class__._onedal_fit,
                    "sklearn": _sklearn_PCA.fit,
                },
                X,
            )
            return self

        def _onedal_fit(self, X, queue=None):
            X = validate_data(
                self,
                X,
                dtype=[np.float64, np.float32],
                ensure_2d=True,
                copy=self.copy,
            )

            if (
                sklearn_check_version("1.5")
                and self._fit_svd_solver == "full"
                and self.svd_solver == "auto"
            ):
                self._fit_svd_solver = "covariance_eigh"
                # warning should only be emitted if to be offloaded to oneDAL
                warn(
                    "Sklearnex always uses `covariance_eigh` solver instead of `full` "
                    "when `svd_solver` parameter is set to `auto` "
                    "for performance purposes."
                )

            # unless the components are explicitly given as an integer, post-processing
            # will set the components having first trained using the minimum size of the
            # input dimensions. This is done in sklearnex and not in the onedal estimator
            n_components = (
                self.n_components
                if isinstance(self.n_components, Integral)
                else min(X.shape)
            )
            onedal_params = {
                "n_components": n_components,
                "is_deterministic": True,
                "method": "svd" if self._fit_svd_solver == "onedal_svd" else "cov",
                "whiten": self.whiten,
            }
            self._onedal_estimator = self.onedal_PCA(**onedal_params)
            self._onedal_estimator.fit(X, queue=queue)

            self.n_samples_ = X.shape[0]
            self.n_features_in_ = X.shape[1]

            # post-process the number of components
            if self.n_components is not None and not isinstance(
                self.n_components, Integral
            ):
                n_components = self._postprocess_n_components()

            # set attributes necessary for calls to transform, will modify
            # self._onedal_estimator, and clear any previous fit models
            self.n_components_ = n_components
            self.components_ = self._onedal_estimator.components_[:n_components, ...]
            self.explained_variance_ = self._onedal_estimator.explained_variance_[
                :n_components
            ]

            # set private mean, as it doesn't need to feed back on the onedal_estimator
            self._mean_ = self._onedal_estimator.mean_

            # set other fit attributes, first by modifying the onedal_estimator
            self._onedal_estimator.singular_values_ = (
                self._onedal_estimator.singular_values_[:n_components]
            )
            self._onedal_esitmator.explained_variance_ratio_ = (
                self._onedal_estimator.explained_variance_ratio_[:n_components]
            )

            self.singular_values_ = self._onedal_estimator.singular_values_
            self.explained_variance_ratio_ = (
                self._onedal_estimator.explained_variance_ratio_
            )

            # calculate the noise variance
            self.noise_variance_ = self._compute_noise_variance(X.shape, xp=xp)

            # return X for use in fit_transform, as it is validated and ready
            return X

        if not sklearn_check_version("1.2"):

            @property
            def n_features_(self):
                return self.n_features_in_

            @n_features_.setter
            def n_features(self, value):
                self.n_features_in_ = value

        @wrap_output_data
        def transform(self, X):
            check_is_fitted(self)
            return dispatch(
                self,
                "transform",
                {
                    "onedal": self.__class__._onedal_transform,
                    "sklearn": _sklearn_PCA.transform,
                },
                X,
            )

        def _onedal_transform(self, X, queue=None):
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                reset=False,
            )

            return self._onedal_estimator.predict(X, queue=queue)

        def _onedal_fit_transform(self, X, queue=None):
            X = self._onedal_fit(X, queue=queue)
            return self._onedal_estimator.predict(X, queue=queue)

        @wrap_output_data
        def fit_transform(self, X):
            return dispatch(
                self,
                "fit_transform",
                {
                    "onedal": self.__class__._onedal_fit_transform,
                    "sklearn": _sklearn_PCA.fit_transform,
                },
                X,
            )

        def inverse_transform(self, X):
            # sklearn does not properly input check inverse_transform using
            # ``validate_data`` (as of sklearn 1.7). Yielding the namespace
            # in this way will conform to sklearn and various inputs without
            # causing issues with dimensionality checks, array api support,
            # etc. evaluated in ``validate_data``. This is a special solution.
            xp = (
                func()
                if (func := getattr(X, "__array_namespace__", None))
                else get_namespace(X)[0]
            )

            mean = self.mean_
            if self.whiten:
                components = (
                    xp.sqrt(self.explained_variance_[:, np.newaxis]) * self.components_
                )
            else:
                components = self.components_

            if not _is_numpy_namespace(xp):
                # Force matching type to input data if possible
                components = xp.asarray(components, device=X.device)
                mean = xp.asarray(mean, device=X.device)

            return X @ components + mean

        # set properties for deleting the onedal_estimator model if:
        # n_components_, components_, means_ or explained_variance_ are
        # changed. This assists in speeding up multiple uses of onedal
        # transform as a model must now only be generated once.

        @property
        def n_components_(self):
            return self._n_components_

        @n_components.setter
        def n_components_(self, value):
            if hasattr(self, "_onedal_estimator"):
                self._onedal_estimator.n_components_ = value
                if hasattr(self._onedal_estimator, "_onedal_model"):
                    del self._onedal_estimator._onedal_model
            self._n_components_ = value

        @property
        def components_(self):
            return self._components_

        @components_.setter
        def components_(self, value):
            if hasattr(self, "_onedal_estimator"):
                self._onedal_estimator.components_ = value
                if hasattr(self._onedal_estimator, "_onedal_model"):
                    del self._onedal_estimator._onedal_model
            self._n_components_ = value

        @property
        def means_(self):
            return self._means_

        @means_.setter
        def means_(self, value):
            if hasattr(self, "_onedal_estimator"):
                self._onedal_estimator.means_ = value
                if hasattr(self._onedal_estimator, "_onedal_model"):
                    del self._onedal_estimator._onedal_model
            self._means_ = value

        @property
        def explained_variance_(self):
            return self._explained_variance_

        @explained_variance_.setter
        def explained_variance_(self, value):
            if hasattr(self, "_onedal_estimator"):
                self._onedal_estimator.explained_variance_ = value
                if hasattr(self._onedal_estimator, "_onedal_model"):
                    del self._onedal_estimator._onedal_model
            self._explained_variance_ = value

        fit.__doc__ = _sklearn_PCA.fit.__doc__
        transform.__doc__ = _sklearn_PCA.transform.__doc__
        fit_transform.__doc__ = _sklearn_PCA.fit_transform.__doc__
        inverse_transform.__doc__ = _sklearn_PCA.inverse_transform.__doc__

else:
    from daal4py.sklearn.decomposition import PCA

    logging.warning(
        "Sklearnex PCA requires oneDAL version >= 2024.1.0 but it was not found"
    )

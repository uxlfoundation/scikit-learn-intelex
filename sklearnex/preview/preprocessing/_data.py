# ==============================================================================
# Copyright Contributors to the oneDAL Project
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

from functools import partial

from sklearn.preprocessing import MaxAbsScaler as _sklearn_MaxAbsScaler
from sklearn.utils.validation import check_array, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import is_sparse, sklearn_check_version
from onedal._device_offload import support_sycl_format
from onedal.basic_statistics import (
    IncrementalBasicStatistics as onedal_IncrementalBasicStatistics,
)

from ..._config import get_config
from ..._device_offload import dispatch, wrap_output_data
from ..._utils import PatchingConditionsChain
from ...base import oneDALEstimator
from ...utils._array_api import enable_array_api, get_namespace
from ...utils.validation import (
    _finite_keyword,
    assert_all_finite,
    validate_data,
)

__check_kwargs = {
    "dtype": None,
    "ensure_2d": False,
    "ensure_min_samples": 0,
    "ensure_min_features": 0,
    "accept_sparse": True,
    _finite_keyword: False,
}

_check_array = partial(check_array, **__check_kwargs)


@enable_array_api
@control_n_jobs(decorated_methods=["fit", "partial_fit", "_onedal_finalize_fit"])
class MaxAbsScaler(oneDALEstimator, _sklearn_MaxAbsScaler):
    __doc__ = _sklearn_MaxAbsScaler.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_MaxAbsScaler._parameter_constraints,
        }

    def __init__(self, *, copy=True):
        self.copy = copy
        self._need_to_finalize = False

    _onedal_incremental_basic_statistics = staticmethod(onedal_IncrementalBasicStatistics)

    def _onedal_supported(self, method_name, *data):
        # The patching condition here checks whether the data is fit for oneDAL.
        # oneDAL's IncrementalBasicStatistics expects dense input in float32/float64 format.
        # MaxAbsScaler in sklearn naturally supports sparse matrices, which creates a scenario
        # for a required fallback to standard sklearn if the input is sparse.

        patching_status = PatchingConditionsChain(
            f"sklearn.preprocessing.{self.__class__.__name__}.{method_name}"
        )
        if method_name in ["fit", "partial_fit"]:
            (X,) = data
            try:
                X_test = _check_array(X)
                assert_all_finite(X_test)  # minimally verify the data
                input_is_finite = True
            except ValueError:
                input_is_finite = False
            patching_status.and_conditions(
                [
                    (not is_sparse(X), "Sparse input is not supported"),
                    (input_is_finite, "Non-finite input is not supported."),
                ]
            )

        return patching_status

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def _onedal_finalize_fit(self, queue=None):
        # This function commits the basic statistics and extracts the values we need to compute scale_.
        # We need the min_ and max_ to compute the maximum absolute value per feature.
        assert hasattr(self, "_onedal_estimator")
        self._onedal_estimator.finalize_fit()

        xp, _ = get_namespace(self._onedal_estimator.min_)

        # Calculate the max absolute scaler
        min_abs = xp.abs(self._onedal_estimator.min_)
        max_abs = xp.abs(self._onedal_estimator.max_)
        self.max_abs_ = xp.maximum(min_abs, max_abs)
        self.scale_ = xp.where(self._max_abs_ == 0, 1.0, self._max_abs_)

        self._need_to_finalize = False

    def _onedal_partial_fit(self, X, queue=None, check_input=True):
        # partial_fit updates the internal _onedal_estimator with the present batch of X.
        first_pass = not hasattr(self, "n_samples_seen_") or self.n_samples_seen_ == 0

        # In sklearn, check_input is used to enforce validation. In combination with use_raw_input config
        # it controls validation of inputs.
        if check_input and not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                reset=first_pass,
                ensure_all_finite=False,
            )

        # We keep track of the samples internally as well to mirror scikit-learn.
        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            self.n_samples_seen_ += X.shape[0]

        if not hasattr(self, "_onedal_estimator"):
            # We specifically only ask for min and max to save overhead since those are the only two
            # statistics required to calculate the max_abs values.
            self._onedal_estimator = self._onedal_incremental_basic_statistics(
                result_options=["min", "max"]
            )

        self._onedal_estimator.partial_fit(X, queue=queue)
        self._need_to_finalize = True

    def _onedal_fit(self, X, queue=None):
        # For a full fit, we must reset the estimator and internal sample count to 0,
        # mimicking a fresh calculation.
        if not get_config()["use_raw_input"]:
            xp, _ = get_namespace(X)
            if sklearn_check_version("1.2"):
                self._validate_params()
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                ensure_all_finite=False,
            )
        else:
            self.n_features_in_ = X.shape[1]

        self.n_samples_seen_ = 0
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator._reset()

        # Execute partial fit just once on the entire dataset.
        self._onedal_partial_fit(X, queue=queue, check_input=False)

        # Must compute the actual class attributes from the oneDAL values.
        self._onedal_finalize_fit()

        return self

    def partial_fit(self, X, y=None):
        # We use dispatch so that validation occurs appropriately. The check_input feature
        # acts identically to sklearn's checking strategy, hence passed through.
        if sklearn_check_version("1.2"):
            self._validate_params()

        # Scikit-Learn implements a check within partial fit natively, so we pass check_input=True implicitly.
        dispatch(
            self,
            "partial_fit",
            {
                "onedal": self.__class__._onedal_partial_fit,
                "sklearn": _sklearn_MaxAbsScaler.partial_fit,
            },
            X,
        )
        return self

    def fit(self, X, y=None):
        if sklearn_check_version("1.2"):
            self._validate_params()

        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_MaxAbsScaler.fit,
            },
            X,
        )
        return self

    # Transform relies completely on standard scikit-learn functionality and does not need to
    # be overridden using oneDAL capabilities as the scale vectors are appropriately populated.
    transform = support_sycl_format(_sklearn_MaxAbsScaler.transform)
    
    # Ensure access to the derived properties without manually calling _onedal_finalize_fit
    # explicitly from the user. We wrap properties that require a finalized state.
    @property
    def max_abs_(self):
        if hasattr(self, "_onedal_estimator") and self._need_to_finalize:
            self._onedal_finalize_fit()
        return self._max_abs_

    @max_abs_.setter
    def max_abs_(self, value):
        self._max_abs_ = value

    @max_abs_.deleter
    def max_abs_(self):
        del self._max_abs_

    @property
    def scale_(self):
        if hasattr(self, "_onedal_estimator") and self._need_to_finalize:
            self._onedal_finalize_fit()
        return self._scale_

    @scale_.setter
    def scale_(self, value):
        self._scale_ = value

    @scale_.deleter
    def scale_(self):
        del self._scale_

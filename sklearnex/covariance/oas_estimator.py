from sklearn.covariance import EmpiricalCovariance
import numpy as np
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import sklearn_check_version

from .._device_offload import dispatch, wrap_output_data

if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data
else:
    validate_data = EmpiricalCovariance._validate_data

@control_n_jobs(decorated_methods=["fit", "score"])
class OASEstimator(EmpiricalCovariance):
    __doc__ = EmpiricalCovariance.__doc__

    def __init__(self, shrinkage=0.1):
        super().__init__()
        self.shrinkage = shrinkage

    def fit(self, X, y=None):
        dispatch(
            self,
            "fit",
            {
                "onedal": self._onedal_fit,
                "sklearn": self._sklearn_fit,
            },
            X,
            y,
        )
        return self

    def _sklearn_fit(self, X, y=None):
        super().fit(X, y)
        mean = np.mean(X, axis=0)
        n_samples, n_features = X.shape
        emp_cov = self.covariance_
        shrinkage = self.shrinkage
        oas_cov = (1 - shrinkage) * emp_cov + shrinkage * np.mean(np.diag(emp_cov)) * np.eye(n_features)
        self.covariance_ = oas_cov

    def _onedal_fit(self, X, y=None, queue=None):
        from onedal.covariance import OASEstimator as onedal_OASEstimator

        use_raw_input = get_config().get("use_raw_input", False) is True
        if not use_raw_input:
            if sklearn_check_version("1.2"):
                self._validate_params()

            if sklearn_check_version("1.0"):
                X = validate_data(
                    self,
                    X,
                    dtype=[np.float64, np.float32],
                    copy=self.copy,
                    force_all_finite=False,
                )
            else:
                X = check_array(
                    X,
                    dtype=[np.float64, np.float32],
                    copy=self.copy,
                    force_all_finite=False,
                )

        onedal_params = {
            "shrinkage": self.shrinkage,
        }

        self._onedal_estimator = onedal_OASEstimator(**onedal_params)
        self._onedal_estimator.fit(X, queue=queue)

        self.covariance_ = self._onedal_estimator.covariance_

    @wrap_output_data
    def score(self, X, y=None):
        check_is_fitted(self)
        return dispatch(
            self,
            "score",
            {
                "onedal": self._onedal_score,
                "sklearn": super().score,
            },
            X,
            y,
        )

    def _onedal_score(self, X, y=None, queue=None):
        from onedal.covariance import OASEstimator as onedal_OASEstimator

        use_raw_input = get_config().get("use_raw_input", False) is True
        if not use_raw_input:
            if sklearn_check_version("1.0"):
                X = validate_data(
                    self,
                    X,
                    dtype=[np.float64, np.float32],
                    copy=self.copy,
                    force_all_finite=False,
                )
            else:
                X = check_array(
                    X,
                    dtype=[np.float64, np.float32],
                    copy=self.copy,
                    force_all_finite=False,
                )

        if not hasattr(self, "_onedal_estimator"):
            onedal_params = {
                "shrinkage": self.shrinkage,
            }
            self._onedal_estimator = onedal_OASEstimator(**onedal_params)
            self._onedal_estimator.fit(X, queue=queue)

        return self._onedal_estimator.score(X, queue=queue)

    def _more_tags(self):
        return {'allow_nan': False}

    fit.__doc__ = EmpiricalCovariance.fit.__doc__
    score.__doc__ = EmpiricalCovariance.score.__doc__

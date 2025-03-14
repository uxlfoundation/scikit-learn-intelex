# ==============================================================================
# Copyright 2020 Intel Corporation
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

from .validation import (
    _daal_check_array,
    _daal_check_X_y,
    check_n_features,
    get_requires_y_tag,
)


def _daal_validate_data(
    self, X, y=None, reset=True, validate_separately=False, **check_params
):
    """Validate input data and set or check the `n_features_in_` attribute.

    Parameters
    ----------
    X : {array-like, sparse matrix, dataframe} of shape \
            (n_samples, n_features)
        The input samples.
    y : array-like of shape (n_samples,), default=None
        The targets. If None, `check_array` is called on `X` and
        `check_X_y` is called otherwise.
    reset : bool, default=True
        Whether to reset the `n_features_in_` attribute.
        If False, the input will be checked for consistency with data
        provided when reset was last True.
    validate_separately : False or tuple of dicts, default=False
        Only used if y is not None.
        If False, call validate_X_y(). Else, it must be a tuple of kwargs
        to be used for calling check_array() on X and y respectively.
    **check_params : kwargs
        Parameters passed to :func:`sklearn.utils.check_array` or
        :func:`sklearn.utils.check_X_y`. Ignored if validate_separately
        is not False.

    Returns
    -------
    out : {ndarray, sparse matrix} or tuple of these
        The validated input. A tuple is returned if `y` is not None.
    """

    if y is None:
        if get_requires_y_tag(self):
            raise ValueError(
                f"This {self.__class__.__name__} estimator "
                f"requires y to be passed, but the target y is None."
            )
        X = _daal_check_array(X, **check_params)
        out = X
    else:
        if validate_separately:
            # We need this because some estimators validate X and y
            # separately, and in general, separately calling check_array()
            # on X and y isn't equivalent to just calling check_X_y()
            # :(
            check_X_params, check_y_params = validate_separately
            X = _daal_check_array(X, **check_X_params)
            y = _daal_check_array(y, **check_y_params)
        else:
            X, y = _daal_check_X_y(X, y, **check_params)
        out = X, y

    if check_params.get("ensure_2d", True):
        check_n_features(self, X, reset=reset)
    return out

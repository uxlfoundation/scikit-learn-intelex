# ==============================================================================
# Copyright contributors to the oneDAL project
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

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2024, "P", 1)):
    from daal4py.sklearn._utils import sklearn_check_version
    from daal4py.sklearn.linear_model.logistic_path import (
        LogisticRegressionCV as _daal4py_LogisticRegressionCV,
    )
    from onedal._device_offload import support_input_format

    from ...linear_model.logistic_regression import (
        LogisticRegression as _sklearnex_LogisticRegression,
    )

    if sklearn_check_version("1.6"):
        from ...base import Tags

    # This is necessary due to how sklearn handles array API inputs
    class LogisticRegressionCV(
        _daal4py_LogisticRegressionCV, _sklearnex_LogisticRegression
    ):
        fit = support_input_format(_daal4py_LogisticRegressionCV.fit)
        predict_proba = _sklearnex_LogisticRegression.predict_proba
        predict_log_proba = _sklearnex_LogisticRegression.predict_log_proba
        decision_function = _sklearnex_LogisticRegression.decision_function

        __doc__ = _daal4py_LogisticRegressionCV.__doc__

        if sklearn_check_version("1.6"):

            def __sklearn_tags__(self) -> "Tags":
                tags = super().__sklearn_tags__()
                tags.onedal_array_api = False
                return tags

# ===============================================================================
# Copyright 2022 Intel Corporation
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

from daal4py.sklearn._utils import sklearn_check_version

from .validation import assert_all_finite

# Not an ideal solution, but this converts the outputs of newer sklearnex tags
# into dicts to match how tags had been used. Someone more clever than me will
# have to find a way of converting older tags into newer ones instead (with
# minimal impact on performance).

if sklearn_check_version("1.6"):
    from sklearn.utils import get_tags as _sklearn_get_tags

    get_tags = lambda estimator: _sklearn_get_tags(estimator).__dict__

else:
    from sklearn.base import BaseEstimator

    get_tags = BaseEstimator._get_tags

__all__ = ["assert_all_finite", "get_tags"]

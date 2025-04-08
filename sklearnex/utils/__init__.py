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

# Not an ideal solution, but this allows for access to the outputs of older
# sklearnex tag dictionaries in a way similar to the sklearn >=1.6 tag
# dataclasses via duck-typing. At some point this must be removed for direct
# use of get_tags in all circumstances, dictated by sklearn support.  This is
# implemented in a way to minimally impact performance.

if sklearn_check_version("1.6"):
    from sklearn.utils import get_tags
else:
    from sklearn.base import BaseEstimator

    class get_tags:
        def __init__(self, obj):
            self._tags = BaseEstimator._get_tags(obj)

        def __getattr__(self, inp):
            return self._tags[inp]


__all__ = ["assert_all_finite", "get_tags"]

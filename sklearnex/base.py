# ===============================================================================
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
# ===============================================================================

from abc import ABC

from daal4py.sklearn._utils import sklearn_check_version


class IntelEstimator(ABC):

    if sklearn_check_version("1.6"):
        # Starting in sklearn 1.6, _more_tags is deprecated. An IntelEstimator
        # is defined to handle the various versioning issues with the tags and
        # with the ongoing rollout of sklearn's array_api support. This will make
        # maintenance easier, and centralize tag changes to a single location.

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.onedal_array_api = False
            return tags

    elif sklearn_check_version("1.3"):

        def _more_tags(self):
            return {"onedal_array_api": False}

    else:
        # array_api_support tag was added in sklearn 1.3 via scikit-learn/scikit-learn#26372
        def _more_tags(self):
            return {"array_api_support": False, "onedal_array_api": False}

    if sklearn_check_version("1.4"):

        def _get_doc_link(self) -> str:
            # This method is meant to generate a clickable doc link for classses
            # in sklearnex that are not part of base scikit-learn. It should be
            # inherited before inheriting from a scikit-learn estimator, otherwise
            # will get overriden by the estimator's original.
            url = super()._get_doc_link()
            if not url:
                module_path, _ = self.__class__.__module__.rsplit(".", 1)
                class_name = self.__class__.__name__
                url = f"https://intel.github.io/scikit-learn-intelex/latest/non-scikit-algorithms.html#{module_path}.{class_name}"
            return url

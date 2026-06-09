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

from dataclasses import dataclass, fields

from sklearn.base import BaseEstimator
from sklearn.utils import Tags as _sklearn_Tags


@dataclass
class Tags(_sklearn_Tags):
    onedal_array_api: bool = False


class oneDALEstimator:
    """Inherited class for all oneDAL-based estimators.

    This class must be inherited before any scikit-learn estimator
    classes (i.e. those which inherit from scikit-learn's BaseEstimator).

    It enables:

      - dispatching to oneDAL
      - html documentation

    Notes
    -----
    Sklearnex-only estimators must inherit this class directly before
    BaseEstimator in order to properly render documentation.  Other
    sklearn classes like Mixins can be before this class in the MRO
    without any impact on functionality.
    """

    def __sklearn_tags__(self) -> Tags:
        # This convention is unnecessarily restrictive with more performant
        # alternatives but it best follows sklearn. Subclasses will now only need
        # to set `onedal_array_api` to True to signify gpu zero-copy support
        # and maintenance is smaller because of underlying sklearn infrastructure
        sktags = super().__sklearn_tags__()
        tag_dict = {field.name: getattr(sktags, field.name) for field in fields(sktags)}
        return Tags(**tag_dict)

    @property
    def _doc_link_module(self) -> str:
        return "sklearnex"

    def _doc_link_url_param_generator(self, *_) -> dict[str, str]:
        return {
            "estimator_module": "sklearn."
            + self.__class__.__module__.rsplit(".", 2)[-2],
            "estimator_name": self.__class__.__name__,
        }

    def _get_doc_link(self) -> str:
        # This method is meant to generate a clickable doc link for classes
        # in sklearnex including those that are not part of base scikit-learn.
        # It should be inherited before inheriting from a scikit-learn estimator.

        mro = self.__class__.__mro__
        # The next object in the Estimators MRO after oneDALEstimator should be
        # the equivalent sklearn estimator, if it is BaseEstimator, it is a
        # sklearnex-only estimator.
        url = BaseEstimator._get_doc_link(self)
        if (
            url
            and oneDALEstimator in mro
            and mro[mro.index(oneDALEstimator) + 1] is BaseEstimator
        ):
            module_path, _ = self.__class__.__module__.rsplit(".", 1)
            class_name = self.__class__.__name__
            url = f"https://uxlfoundation.github.io/scikit-learn-intelex/latest/non-scikit-algorithms.html#{module_path}.{class_name}"

        return url

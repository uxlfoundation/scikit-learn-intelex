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
from abc import ABC, abstractmethod

from sklearn.utils._array_api import get_namespace


# Adds common methods to create oneDAL objects after a fallback
class _BaseLinearModel(ABC):
    @abstractmethod
    def _initialize_onedal_estimator(self, override_fit_intercept: bool = False) -> None:
        raise NotImplementedError()

    def _get_fit_intercept(self, override_fit_intercept: bool = False) -> bool:
        if not override_fit_intercept:
            return self.fit_intercept
        else:
            if isinstance(self.intercept_, float):
                return self.intercept_ != 0.0
            else:
                xp, _ = get_namespace(self.coef_)
                return bool(xp.all(self.intercept_ != 0))

    def _initialize_onedal_estimator_from_coefs(self) -> None:
        xp, _ = get_namespace(self.coef_)
        self._initialize_onedal_estimator(override_fit_intercept=True)
        self._onedal_estimator._create_model(self.coef_, self.intercept_, xp)

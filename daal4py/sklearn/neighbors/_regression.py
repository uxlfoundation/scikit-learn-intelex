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

# daal4py KNN regression scikit-learn-compatible classes

from sklearn.base import RegressorMixin
from sklearn.neighbors._regression import KNeighborsRegressor as BaseKNeighborsRegressor

from .._utils import sklearn_check_version
from ..utils.validation import check_feature_names
from ._base import KNeighborsMixin, NeighborsBase

if not sklearn_check_version("1.2"):
    from sklearn.neighbors._base import _check_weights

from sklearn.utils.validation import _deprecate_positional_args


class KNeighborsRegressor(KNeighborsMixin, RegressorMixin, NeighborsBase):
    __doc__ = BaseKNeighborsRegressor.__doc__

    @_deprecate_positional_args
    def __init__(
        self,
        n_neighbors=5,
        *,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
        metric_params=None,
        n_jobs=None,
        **kwargs,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
            **kwargs,
        )
        self.weights = weights

    def _more_tags(self):
        return BaseKNeighborsRegressor._more_tags(self)

    def fit(self, X, y):
        return NeighborsBase._fit(self, X, y)

    def predict(self, X):
        check_feature_names(self, X, reset=False)
        return BaseKNeighborsRegressor.predict(self, X)

    fit.__doc__ = BaseKNeighborsRegressor.fit.__doc__
    predict.__doc__ = BaseKNeighborsRegressor.predict.__doc__

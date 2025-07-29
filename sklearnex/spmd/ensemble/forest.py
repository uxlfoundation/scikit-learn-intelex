# ==============================================================================
# Copyright 2023 Intel Corporation
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

from daal4py.sklearn._utils import sklearn_check_version
from onedal.spmd.ensemble import RandomForestClassifier as onedal_RandomForestClassifier
from onedal.spmd.ensemble import RandomForestRegressor as onedal_RandomForestRegressor

from ...ensemble import RandomForestClassifier as RandomForestClassifier_Batch
from ...ensemble import RandomForestRegressor as RandomForestRegressor_Batch


def local_trees_wrapper(func):
    def new_factory(self, **params):
        params["local_trees_mode"] = self.local_trees_mode
        return func(self, **params)

    return new_factory


class RandomForestClassifier(RandomForestClassifier_Batch):
    __doc__ = RandomForestClassifier_Batch.__doc__
    _onedal_factory = onedal_RandomForestClassifier

    # Wrap _onedal_factory to suport local_trees_mode parameter
    @property
    def _onedal_factory(self):
        return local_trees_wrapper(type(self)._onedal_factory)

    # Init constructor with local_trees_mode parameter but pass to parent
    # class without (to maintain scikit-learn estimator compatibility)
    if sklearn_check_version("1.4"):

        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
            max_bins=256,
            min_bin_size=1,
            local_trees_mode=False,
        ):
            self.local_trees_mode = local_trees_mode
            super().__init__(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                monotonic_cst=monotonic_cst,
                max_bins=max_bins,
                min_bin_size=min_bin_size,
            )

    else:

        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features="sqrt" if sklearn_check_version("1.1") else "auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None,
            max_bins=256,
            min_bin_size=1,
            local_trees_mode=False,
        ):
            self.local_trees_mode = local_trees_mode
            super().__init__(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                max_bins=max_bins,
                min_bin_size=min_bin_size,
            )

    def _onedal_cpu_supported(self, method_name, *data):
        # TODO:
        # check which methods supported SPMD interface on CPU.
        ready = super()._onedal_cpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready

    def _onedal_gpu_supported(self, method_name, *data):
        ready = super()._onedal_gpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready


class RandomForestRegressor(RandomForestRegressor_Batch):
    __doc__ = RandomForestRegressor_Batch.__doc__
    _onedal_factory = onedal_RandomForestRegressor

    # Wrap _onedal_factory to suport local_trees_mode parameter
    @property
    def _onedal_factory(self):
        return local_trees_wrapper(type(self)._onedal_factory)

    # Init constructor with local_trees_mode parameter but pass to parent
    # class without (to maintain scikit-learn estimator compatibility)
    if sklearn_check_version("1.4"):

        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            monotonic_cst=None,
            max_bins=256,
            min_bin_size=1,
            local_trees_mode=False,
        ):
            self.local_trees_mode = local_trees_mode
            super().__init__(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                monotonic_cst=monotonic_cst,
                max_bins=max_bins,
                min_bin_size=min_bin_size,
            )

    else:

        def __init__(
            self,
            n_estimators=100,
            *,
            criterion="squared_error",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=1.0 if sklearn_check_version("1.1") else "auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            max_bins=256,
            min_bin_size=1,
            local_trees_mode=False,
        ):
            self.local_trees_mode = local_trees_mode
            super().__init__(
                n_estimators=n_estimators,
                criterion=criterion,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                min_weight_fraction_leaf=min_weight_fraction_leaf,
                max_features=max_features,
                max_leaf_nodes=max_leaf_nodes,
                min_impurity_decrease=min_impurity_decrease,
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                ccp_alpha=ccp_alpha,
                max_samples=max_samples,
                max_bins=max_bins,
                min_bin_size=min_bin_size,
            )

    def _onedal_cpu_supported(self, method_name, *data):
        # TODO:
        # check which methods supported SPMD interface on CPU.
        ready = super()._onedal_cpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready

    def _onedal_gpu_supported(self, method_name, *data):
        ready = super()._onedal_gpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready

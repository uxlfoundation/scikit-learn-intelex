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

import math
import numbers
from abc import ABC, abstractmethod

from daal4py.sklearn._utils import daal_check_version
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM
from sklearnex import get_hyperparameters

from ..datatypes import from_table, to_table


class BaseForest(ABC):
    @abstractmethod
    def __init__(
        self,
        n_estimators,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        min_impurity_decrease,
        min_impurity_split,
        bootstrap,
        random_state,
        observations_per_tree_fraction,
        max_bins,
        min_bin_size,
        infer_mode,
        splitter_mode,
        voting_mode,
        error_metric_mode,
        variable_importance_mode,
        algorithm,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.observations_per_tree_fraction = observations_per_tree_fraction
        self.max_bins = max_bins
        self.min_bin_size = min_bin_size
        self.infer_mode = infer_mode
        self.splitter_mode = splitter_mode
        self.voting_mode = voting_mode
        self.error_metric_mode = error_metric_mode
        self.variable_importance_mode = variable_importance_mode
        self.algorithm = algorithm

    @abstractmethod
    def train(self, *args, **kwargs): ...

    @abstractmethod
    def infer(self, *args, **kwargs): ...

    def _to_absolute_max_features(self, n_features):
        # This functionality is not used when the onedal estimator is called
        # by the sklearnex estimator, it is superseded by the
        # `_to_absolute_max_features` function which handles additional
        # sklearn conformance. This is kept as accessing the default values
        # (Classification: sqrt(p), Regression: p/3) is not possible with
        # current implementation in forest.cpp
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, str):
            return max(1, int(getattr(math, self.max_features)(n_features)))
        elif isinstance(self.max_features, numbers.Integral):
            return self.max_features
        elif self.max_features > 0.0:
            return max(1, int(self.max_features * n_features))
        return 0

    def _get_onedal_params(self, data):
        n_samples, n_features = data.shape

        min_observations_in_leaf_node = (
            self.min_samples_leaf
            if isinstance(self.min_samples_leaf, numbers.Integral)
            else int(math.ceil(self.min_samples_leaf * n_samples))
        )

        min_observations_in_split_node = (
            self.min_samples_split
            if isinstance(self.min_samples_split, numbers.Integral)
            else int(math.ceil(self.min_samples_split * n_samples))
        )

        onedal_params = {
            "fptype": data.dtype,
            "method": self.algorithm,
            "observations_per_tree_fraction": self.observations_per_tree_fraction,
            "impurity_threshold": float(
                0.0 if self.min_impurity_split is None else self.min_impurity_split
            ),
            "min_weight_fraction_in_leaf_node": self.min_weight_fraction_leaf,
            "min_impurity_decrease_in_split_node": self.min_impurity_decrease,
            "tree_count": int(self.n_estimators),
            "features_per_node": self._to_absolute_max_features(n_features),
            "max_tree_depth": int(0 if self.max_depth is None else self.max_depth),
            "min_observations_in_leaf_node": min_observations_in_leaf_node,
            "min_observations_in_split_node": min_observations_in_split_node,
            "max_leaf_nodes": (0 if self.max_leaf_nodes is None else self.max_leaf_nodes),
            "max_bins": self.max_bins,
            "min_bin_size": self.min_bin_size,
            "seed": self.random_state,
            "memory_saving_mode": False,
            "bootstrap": bool(self.bootstrap),
            "error_metric_mode": self.error_metric_mode,
            "variable_importance_mode": self.variable_importance_mode,
            "class_count": self.class_count_,  # used in classification only
            "infer_mode": self.infer_mode,  # used in classification only
            "voting_mode": self.voting_mode,  # used in classification only
        }

        if daal_check_version((2023, "P", 101)):
            onedal_params["splitter_mode"] = self.splitter_mode
        return onedal_params

    @supports_queue
    def fit(self, X, y, sample_weight=None, class_count=0, queue=None):
        self.class_count_ = class_count
        if sample_weight is not None and len(sample_weight) > 0:
            data = (X, y, sample_weight)
        else:
            data = (X, y)
        data = to_table(*data, queue=queue)
        params = self._get_onedal_params(data[0])
        train_result = self.train(params, *data)

        self._onedal_model = train_result.model

        # set attributes related to the various oob metric modes
        # naming scheme of mode matches the attribute of the result
        # object. See decision forest documentation for settings.
        if "none" not in self.error_metric_mode:
            for i in self.error_metric_mode.replace(
                "out_of_bag_error_", "oob_err_"
            ).split("|"):
                setattr(self, i + "_", from_table(getattr(train_result, i), like=X))

        if self.variable_importance_mode != "none":
            self.var_importance_ = from_table(train_result.var_importance, like=X)[0, :]

        return self

    @supports_queue
    def predict(self, X, queue=None):
        hparams = get_hyperparameters("decision_forest", "infer")

        queue = QM.get_global_queue()
        X_table = to_table(X, queue=queue)
        params = self._get_onedal_params(X_table)
        if hparams is not None and not hparams.is_default:
            result = self.infer(params, hparams.backend, self._onedal_model, X_table)
        else:
            result = self.infer(params, self._onedal_model, X_table)

        y = from_table(result.responses, like=X)[:, 0]
        return y


class ForestClassifier(BaseForest):

    @bind_default_backend("decision_forest.classification")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("decision_forest.classification")
    def infer(self, *args, **kwargs): ...

    @supports_queue
    def predict_proba(self, X, queue=None):
        hparams = get_hyperparameters("decision_forest", "infer")
        X_table = to_table(X, queue=queue)
        params = self._get_onedal_params(X_table)
        params["infer_mode"] = "class_probabilities"

        if hparams is not None and not hparams.is_default:
            result = self.infer(params, hparams.backend, self._onedal_model, X_table)
        else:
            result = self.infer(params, self._onedal_model, X_table)

        return from_table(result.probabilities, like=X)


class ForestRegressor(BaseForest):
    @bind_default_backend("decision_forest.regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("decision_forest.regression")
    def infer(self, *args, **kwargs): ...


class RandomForestClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        random_state=None,
        observations_per_tree_fraction=1.0,
        max_bins=256,
        min_bin_size=1,
        infer_mode="class_responses",
        splitter_mode="best",
        voting_mode="weighted",
        error_metric_mode="none",
        variable_importance_mode="none",
        algorithm="hist",
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            random_state=random_state,
            observations_per_tree_fraction=observations_per_tree_fraction,
            max_bins=max_bins,
            min_bin_size=min_bin_size,
            infer_mode=infer_mode,
            splitter_mode=splitter_mode,
            voting_mode=voting_mode,
            error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode,
            algorithm=algorithm,
        )

    @bind_default_backend("decision_forest.classification")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("decision_forest.classification")
    def infer(self, *args, **kwargs): ...


class RandomForestRegressor(ForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        random_state=None,
        observations_per_tree_fraction=1.0,
        max_bins=256,
        min_bin_size=1,
        infer_mode="class_responses",  # not used (see forest.cpp)
        splitter_mode="best",
        voting_mode="weighted",  # not used (see forest.cpp)
        error_metric_mode="none",
        variable_importance_mode="none",
        algorithm="hist",
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            random_state=random_state,
            observations_per_tree_fraction=observations_per_tree_fraction,
            max_bins=max_bins,
            min_bin_size=min_bin_size,
            infer_mode=infer_mode,
            splitter_mode=splitter_mode,
            voting_mode=voting_mode,
            error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode,
            algorithm=algorithm,
        )


class ExtraTreesClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=False,
        random_state=None,
        observations_per_tree_fraction=1.0,
        max_bins=256,
        min_bin_size=1,
        infer_mode="class_responses",
        splitter_mode="random",
        voting_mode="weighted",
        error_metric_mode="none",
        variable_importance_mode="none",
        algorithm="hist",
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            random_state=random_state,
            observations_per_tree_fraction=observations_per_tree_fraction,
            max_bins=max_bins,
            min_bin_size=min_bin_size,
            infer_mode=infer_mode,
            splitter_mode=splitter_mode,
            voting_mode=voting_mode,
            error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode,
            algorithm=algorithm,
        )


class ExtraTreesRegressor(ForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=False,
        random_state=None,
        observations_per_tree_fraction=1.0,
        max_bins=256,
        min_bin_size=1,
        infer_mode="class_responses",  # not used (see forest.cpp)
        splitter_mode="random",
        voting_mode="weighted",  # not used (see forest.cpp)
        error_metric_mode="none",
        variable_importance_mode="none",
        algorithm="hist",
    ):
        super().__init__(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            random_state=random_state,
            observations_per_tree_fraction=observations_per_tree_fraction,
            max_bins=max_bins,
            min_bin_size=min_bin_size,
            infer_mode=infer_mode,
            splitter_mode=splitter_mode,
            voting_mode=voting_mode,
            error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode,
            algorithm=algorithm,
        )

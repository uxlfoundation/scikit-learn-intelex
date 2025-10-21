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
from abc import ABCMeta, abstractmethod

from daal4py.sklearn._utils import daal_check_version
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM
from sklearnex import get_hyperparameters

from .._config import _get_config
from ..common._estimator_checks import _check_is_fitted
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace


class BaseForest(metaclass=ABCMeta):
    @abstractmethod
    def __init__(
        self,
        n_estimators,
        criterion,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        min_weight_fraction_leaf,
        max_features,
        max_leaf_nodes,
        min_impurity_decrease,
        min_impurity_split,
        bootstrap,
        oob_score,
        random_state,
        warm_start,
        class_weight,
        ccp_alpha,
        max_samples,
        max_bins,
        min_bin_size,
        infer_mode,
        splitter_mode,
        voting_mode,
        error_metric_mode,
        variable_importance_mode,
        algorithm,
        **kwargs,
    ):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.random_state = random_state
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.max_samples = max_samples
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.ccp_alpha = ccp_alpha
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
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, str):
            return max(1, int(getattr(math, self.max_features)(n_features)))
        elif isinstance(self.max_features, numbers.Integral):
            return self.max_features
        elif self.max_features > 0.0:
            return max(1, int(self.max_features * n_features))
        return 0

    def _get_observations_per_tree_fraction(self, n_samples, max_samples):
        if max_samples is None:
            return 1.0

        elif isinstance(max_samples, numbers.Integral):
            if not (1 <= max_samples <= n_samples):
                msg = "`max_samples` must be in range 1 to {} but got value {}"
                raise ValueError(msg.format(n_samples, max_samples))
            return max(float(max_samples / n_samples), 1 / n_samples)

        elif isinstance(max_samples, numbers.Real):
            return max(float(max_samples), 1 / n_samples)

        msg = "`max_samples` should be int or float, but got type '{}'"
        raise TypeError(msg.format(type(max_samples)))

    def _get_onedal_params(self, data):
        n_samples, n_features = data.shape

        self.observations_per_tree_fraction = (
            self._get_observations_per_tree_fraction(
                n_samples=n_samples, max_samples=self.max_samples
            )
            if bool(self.bootstrap)
            else 1.0
        )

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
            "infer_mode": self.infer_mode,
            "voting_mode": self.voting_mode,
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
            "class_count": self.class_count_,
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

        return train_result

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

    @supports_queue
    def predict_proba(self, X, queue=None):
        hparams = get_hyperparameters("decision_forest", "infer")
        X = to_table(X, queue=queue)
        params = self._get_onedal_params(X)
        params["infer_mode"] = "class_probabilities"

        if hparams is not None and not hparams.is_default:
            result = self.infer(params, hparams.backend, self._onedal_model, X)
        else:
            result = self.infer(params, self._onedal_model, X)

        # TODO: fix probabilities out of [0, 1] interval on oneDAL side
        pred = from_table(result.probabilities)
        pred[pred > 1.0] = 1.0
        pred[pred < 0.0] = 0.0
        return pred

    @supports_queue
    def fit(self, X, y, sample_weight=None, class_count=0, queue=None):
        train_result = super().fit(
            X, y, sample_weight=sample_weight, class_count=class_count, queue=queue
        )

        self.oob_score_ = from_table(train_result.oob_err_accuracy, like=X)
        self.oob_decision_function_ = from_table(
            train_result.oob_err_decision_function, like=X
        )

        return self


class ForestRegressor(BaseForest):

    @supports_queue
    def fit(self, X, y, sample_weight=None, class_count=0, queue=None):
        train_result = super().fit(
            X, y, sample_weight=sample_weight, class_count=class_count, queue=queue
        )

        self.oob_score_ = from_table(train_result.oob_err_r2, like=X)
        self.oob_prediction_ = from_table(train_result.oob_err_prediction, like=X)

        return self


class RandomForestClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
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
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
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
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        random_state=None,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
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
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            max_bins=max_bins,
            min_bin_size=min_bin_size,
            infer_mode=infer_mode,
            splitter_mode=splitter_mode,
            voting_mode=voting_mode,
            error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode,
            algorithm=algorithm,
        )

    @bind_default_backend("decision_forest.regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("decision_forest.regression")
    def infer(self, *args, **kwargs): ...


class ExtraTreesClassifier(ForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=False,
        oob_score=False,
        random_state=None,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
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
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
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


class ExtraTreesRegressor(ForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=False,
        oob_score=False,
        random_state=None,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
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
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            random_state=random_state,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            max_bins=max_bins,
            min_bin_size=min_bin_size,
            infer_mode=infer_mode,
            splitter_mode=splitter_mode,
            voting_mode=voting_mode,
            error_metric_mode=error_metric_mode,
            variable_importance_mode=variable_importance_mode,
            algorithm=algorithm,
        )

    @bind_default_backend("decision_forest.regression")
    def train(self, *args, **kwargs): ...

    @bind_default_backend("decision_forest.regression")
    def infer(self, *args, **kwargs): ...

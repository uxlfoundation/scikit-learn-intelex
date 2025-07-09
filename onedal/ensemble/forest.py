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

import numbers
import warnings
from abc import ABCMeta, abstractmethod
from math import ceil

import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.utils import check_random_state

from daal4py.sklearn._utils import daal_check_version
from onedal._device_offload import supports_queue
from onedal.common._backend import bind_default_backend
from onedal.utils import _sycl_queue_manager as QM
from sklearnex import get_hyperparameters

from .._config import _get_config
from ..common._estimator_checks import _check_is_fitted
from ..common._mixin import ClassifierMixin, RegressorMixin
from ..datatypes import from_table, to_table
from ..utils._array_api import _get_sycl_namespace
from ..utils.validation import (
    _check_array,
    _check_n_features,
    _check_X_y,
    _column_or_1d,
    _validate_targets,
)


class BaseForest(BaseEnsemble, metaclass=ABCMeta):
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
            return max(1, int(getattr(np, self.max_features)(n_features)))
        elif isinstance(self.max_features, (numbers.Integral, np.integer)):
            return self.max_features
        elif self.max_features > 0.0:
            return max(1, int(self.max_features * n_features))
        return 0

    def _get_observations_per_tree_fraction(self, n_samples, max_samples):
        if max_samples is None:
            return 1.0

        if isinstance(max_samples, numbers.Integral):
            if not (1 <= max_samples <= n_samples):
                msg = "`max_samples` must be in range 1 to {} but got value {}"
                raise ValueError(msg.format(n_samples, max_samples))
            return max(float(max_samples / n_samples), 1 / n_samples)

        if isinstance(max_samples, numbers.Real):
            return max(float(max_samples), 1 / n_samples)

        msg = "`max_samples` should be int or float, but got type '{}'"
        raise TypeError(msg.format(type(max_samples)))

    def _get_onedal_params(self, data):
        n_samples, n_features = data.shape

        self.observations_per_tree_fraction = self._get_observations_per_tree_fraction(
            n_samples=n_samples, max_samples=self.max_samples
        )
        self.observations_per_tree_fraction = (
            self.observations_per_tree_fraction if bool(self.bootstrap) else 1.0
        )

        if not self.bootstrap and self.max_samples is not None:
            raise ValueError(
                "`max_sample` cannot be set if `bootstrap=False`. "
                "Either switch to `bootstrap=True` or set "
                "`max_sample=None`."
            )
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available" " if bootstrap=True")

        min_observations_in_leaf_node = (
            self.min_samples_leaf
            if isinstance(self.min_samples_leaf, numbers.Integral)
            else int(ceil(self.min_samples_leaf * n_samples))
        )

        min_observations_in_split_node = (
            self.min_samples_split
            if isinstance(self.min_samples_split, numbers.Integral)
            else int(ceil(self.min_samples_split * n_samples))
        )

        rs = check_random_state(self.random_state)
        seed = rs.randint(0, np.iinfo("i").max)

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
            "seed": seed,
            "memory_saving_mode": False,
            "bootstrap": bool(self.bootstrap),
            "error_metric_mode": self.error_metric_mode,
            "variable_importance_mode": self.variable_importance_mode,
        }
        if isinstance(self, ClassifierMixin):
            onedal_params["class_count"] = (
                0 if self.classes_ is None else len(self.classes_)
            )
        if daal_check_version((2023, "P", 101)):
            onedal_params["splitter_mode"] = self.splitter_mode
        return onedal_params

    def _check_parameters(self):
        if isinstance(self.min_samples_leaf, numbers.Integral):
            if not 1 <= self.min_samples_leaf:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
        else:  # float
            if not 0.0 < self.min_samples_leaf <= 0.5:
                raise ValueError(
                    "min_samples_leaf must be at least 1 "
                    "or in (0, 0.5], got %s" % self.min_samples_leaf
                )
        if isinstance(self.min_samples_split, numbers.Integral):
            if not 2 <= self.min_samples_split:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the integer %s" % self.min_samples_split
                )
        else:  # float
            if not 0.0 < self.min_samples_split <= 1.0:
                raise ValueError(
                    "min_samples_split must be an integer "
                    "greater than 1 or a float in (0.0, 1.0]; "
                    "got the float %s" % self.min_samples_split
                )
        if not 0 <= self.min_weight_fraction_leaf <= 0.5:
            raise ValueError("min_weight_fraction_leaf must in [0, 0.5]")
        if self.min_impurity_split is not None:
            warnings.warn(
                "The min_impurity_split parameter is deprecated. "
                "Its default value has changed from 1e-7 to 0 in "
                "version 0.23, and it will be removed in 0.25. "
                "Use the min_impurity_decrease parameter instead.",
                FutureWarning,
            )

            if self.min_impurity_split < 0.0:
                raise ValueError(
                    "min_impurity_split must be greater than " "or equal to 0"
                )
        if self.min_impurity_decrease < 0.0:
            raise ValueError(
                "min_impurity_decrease must be greater than " "or equal to 0"
            )
        if self.max_leaf_nodes is not None:
            if not isinstance(self.max_leaf_nodes, numbers.Integral):
                raise ValueError(
                    "max_leaf_nodes must be integral number but was "
                    "%r" % self.max_leaf_nodes
                )
            if self.max_leaf_nodes < 2:
                raise ValueError(
                    ("max_leaf_nodes {0} must be either None " "or larger than 1").format(
                        self.max_leaf_nodes
                    )
                )
        if isinstance(self.max_bins, numbers.Integral):
            if not 2 <= self.max_bins:
                raise ValueError("max_bins must be at least 2, got %s" % self.max_bins)
        else:
            raise ValueError(
                "max_bins must be integral number but was " "%r" % self.max_bins
            )
        if isinstance(self.min_bin_size, numbers.Integral):
            if not 1 <= self.min_bin_size:
                raise ValueError(
                    "min_bin_size must be at least 1, got %s" % self.min_bin_size
                )
        else:
            raise ValueError(
                "min_bin_size must be integral number but was " "%r" % self.min_bin_size
            )

    def _validate_targets(self, y, dtype):
        self.class_weight_ = None
        self.classes_ = None
        return _column_or_1d(y, warn=True).astype(dtype, copy=False)

    def _get_sample_weight(self, sample_weight, X):
        sample_weight = np.asarray(sample_weight, dtype=X.dtype).ravel()

        sample_weight = _check_array(
            sample_weight, accept_sparse=False, ensure_2d=False, dtype=X.dtype, order="C"
        )

        if sample_weight.size != X.shape[0]:
            raise ValueError(
                "sample_weight and X have incompatible shapes: "
                "%r vs %r\n"
                "Note: Sparse matrices cannot be indexed w/"
                "boolean masks (use `indices=True` in CV)."
                % (sample_weight.shape, X.shape)
            )

        return sample_weight

    def _fit(self, X, y, sample_weight):
        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, xp, _ = _get_sycl_namespace(X)

        if not use_raw_input:
            X, y = _check_X_y(
                X,
                y,
                dtype=[np.float64, np.float32],
                force_all_finite=True,
                accept_sparse="csr",
            )
            y = self._validate_targets(y, X.dtype)
        else:
            if sua_iface is not None:
                queue = X.sycl_queue
            # try catch needed for raw_inputs + array_api data where unlike
            # numpy the way to yield unique values is via `unique_values`
            # This should be removed when refactored for gpu zero-copy
            try:
                self.classes_ = xp.unique(y)
            except AttributeError:
                self.classes_ = xp.unique_values(y)

        self.n_features_in_ = X.shape[1]

        if sample_weight is not None and len(sample_weight) > 0:
            if not use_raw_input:
                sample_weight = self._get_sample_weight(sample_weight, X)
            data = (X, y, sample_weight)
        else:
            data = (X, y)
        data = to_table(*data, queue=QM.get_global_queue())
        params = self._get_onedal_params(data[0])
        train_result = self.train(params, *data)

        self._onedal_model = train_result.model

        if self.oob_score:
            if isinstance(self, ClassifierMixin):
                self.oob_score_ = from_table(train_result.oob_err_accuracy).item()
                self.oob_decision_function_ = from_table(
                    train_result.oob_err_decision_function
                )
                if xp.any(self.oob_decision_function_ == 0):
                    warnings.warn(
                        "Some inputs do not have OOB scores. This probably means "
                        "too few trees were used to compute any reliable OOB "
                        "estimates.",
                        UserWarning,
                    )
            else:
                self.oob_score_ = from_table(train_result.oob_err_r2).item()
                self.oob_prediction_ = from_table(
                    train_result.oob_err_prediction
                ).reshape(-1)
                if np.any(self.oob_prediction_ == 0):
                    warnings.warn(
                        "Some inputs do not have OOB scores. This probably means "
                        "too few trees were used to compute any reliable OOB "
                        "estimates.",
                        UserWarning,
                    )

        return self

    def _create_model(self, module):
        # TODO:
        # update error msg.
        raise NotImplementedError("Creating model is not supported.")

    def _predict(self, X, hparams=None):
        _check_is_fitted(self)

        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, xp, _ = _get_sycl_namespace(X)

        # All data should use the same sycl queue
        if use_raw_input and sua_iface is not None:
            queue = X.sycl_queue

        if not use_raw_input:
            X = _check_array(
                X,
                dtype=[np.float64, np.float32],
                force_all_finite=True,
                accept_sparse=False,
            )
            _check_n_features(self, X, False)

        model = self._onedal_model
        queue = QM.get_global_queue()
        X = to_table(X, queue=queue)
        params = self._get_onedal_params(X)
        if hparams is not None and not hparams.is_default:
            result = self.infer(params, hparams.backend, model, X)
        else:
            result = self.infer(params, model, X)

        y = from_table(result.responses, sua_iface=sua_iface, sycl_queue=queue, xp=xp)
        return y

    def _predict_proba(self, X, hparams=None):
        _check_is_fitted(self)
        use_raw_input = _get_config().get("use_raw_input", False) is True
        sua_iface, xp, _ = _get_sycl_namespace(X)

        # All data should use the same sycl queue
        if use_raw_input and sua_iface is not None:
            queue = X.sycl_queue
        else:
            queue = QM.get_global_queue()

        if not use_raw_input:
            X = _check_array(
                X,
                dtype=[np.float64, np.float32],
                force_all_finite=True,
                accept_sparse=False,
            )
            _check_n_features(self, X, False)
        X = to_table(X, queue=queue)
        params = self._get_onedal_params(X)
        params["infer_mode"] = "class_probabilities"

        model = self._onedal_model
        if hparams is not None and not hparams.is_default:
            result = self.infer(params, hparams.backend, model, X)
        else:
            result = self.infer(params, model, X)

        # TODO: fix probabilities out of [0, 1] interval on oneDAL side
        pred = from_table(result.probabilities)
        return pred.clip(0.0, 1.0)


class RandomForestClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
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
        **kwargs,
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

    def _validate_targets(self, y, dtype):
        y, self.class_weight_, self.classes_ = _validate_targets(
            y, self.class_weight, dtype
        )

        # Decapsulate classes_ attributes
        # TODO:
        # align with `n_classes_` and `classes_` attr with daal4py implementations.
        # if hasattr(self, "classes_"):
        #    self.n_classes_ = self.classes_
        return y

    @supports_queue
    def fit(self, X, y, sample_weight=None, queue=None):
        return self._fit(X, y, sample_weight)

    @supports_queue
    def predict(self, X, queue=None):
        _, xp, _ = _get_sycl_namespace(X)
        hparams = get_hyperparameters("decision_forest", "infer")
        pred = xp.reshape(self._predict(X, hparams), -1)

        try:
            return xp.take(
                xp.asarray(self.classes_, device=pred.sycl_queue),
                xp.astype(xp.reshape(pred, (-1,)), xp.int64),
            )
        except AttributeError:
            return np.take(self.classes_, pred.ravel().astype(np.int64, casting="unsafe"))

    @supports_queue
    def predict_proba(self, X, queue=None):
        hparams = get_hyperparameters("decision_forest", "infer")

        return super()._predict_proba(X, hparams)


class RandomForestRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta):
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
        **kwargs,
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

    @supports_queue
    def fit(self, X, y, sample_weight=None, queue=None):
        if sample_weight is not None:
            if hasattr(sample_weight, "__array__"):
                sample_weight[sample_weight == 0.0] = 1.0
            sample_weight = [sample_weight]
        return self._fit(X, y, sample_weight)

    @supports_queue
    def predict(self, X, queue=None):
        _, xp, _ = _get_sycl_namespace(X)
        return xp.reshape(self._predict(X), -1)


class ExtraTreesClassifier(ClassifierMixin, BaseForest, metaclass=ABCMeta):
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
        **kwargs,
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

    def _validate_targets(self, y, dtype):
        y, self.class_weight_, self.classes_ = _validate_targets(
            y, self.class_weight, dtype
        )

        # Decapsulate classes_ attributes
        # TODO:
        # align with `n_classes_` and `classes_` attr with daal4py implementations.
        # if hasattr(self, "classes_"):
        #    self.n_classes_ = self.classes_
        return y

    @supports_queue
    def fit(self, X, y, sample_weight=None, queue=None):
        return self._fit(X, y, sample_weight)

    @supports_queue
    def predict(self, X, queue=None):
        pred = self._predict(X)

        return np.take(self.classes_, pred.ravel().astype(np.int64, casting="unsafe"))

    @supports_queue
    def predict_proba(self, X, queue=None):
        return super()._predict_proba(X)


class ExtraTreesRegressor(RegressorMixin, BaseForest, metaclass=ABCMeta):
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
        **kwargs,
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

    @supports_queue
    def fit(self, X, y, sample_weight=None, queue=None):
        if sample_weight is not None:
            if hasattr(sample_weight, "__array__"):
                sample_weight[sample_weight == 0.0] = 1.0
            sample_weight = [sample_weight]
        return self._fit(X, y, sample_weight)

    @supports_queue
    def predict(self, X, queue=None):
        return self._predict(X).ravel()

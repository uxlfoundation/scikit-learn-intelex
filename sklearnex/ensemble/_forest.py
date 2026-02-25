# ==============================================================================
# Copyright 2021 Intel Corporation
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
import warnings
from abc import ABC
from functools import partial

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.ensemble import ExtraTreesClassifier as _sklearn_ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor as _sklearn_ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier as _sklearn_RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor as _sklearn_RandomForestRegressor
from sklearn.ensemble._forest import ForestClassifier as _sklearn_ForestClassifier
from sklearn.ensemble._forest import ForestRegressor as _sklearn_ForestRegressor
from sklearn.ensemble._forest import _get_n_samples_bootstrap
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics import accuracy_score, r2_score
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ExtraTreeClassifier,
    ExtraTreeRegressor,
)
from sklearn.tree._tree import Tree
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_array, check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import (
    check_tree_nodes,
    daal_check_version,
    is_sparse,
    sklearn_check_version,
)
from onedal._device_offload import support_input_format
from onedal.ensemble import ExtraTreesClassifier as onedal_ExtraTreesClassifier
from onedal.ensemble import ExtraTreesRegressor as onedal_ExtraTreesRegressor
from onedal.ensemble import RandomForestClassifier as onedal_RandomForestClassifier
from onedal.ensemble import RandomForestRegressor as onedal_RandomForestRegressor
from onedal.primitives import get_tree_state_cls, get_tree_state_reg
from onedal.utils.validation import _num_features

from .._config import get_config
from .._device_offload import dispatch, wrap_output_data
from .._utils import PatchingConditionsChain, register_hyperparameters
from ..base import oneDALEstimator
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.class_weight import _compute_class_weight
from ..utils.validation import (
    _check_sample_weight,
    _finite_keyword,
    assert_all_finite,
    validate_data,
)

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import Interval


__check_kwargs = {
    "dtype": None,
    "ensure_2d": False,
    "ensure_min_samples": 0,
    "ensure_min_features": 0,
    "accept_sparse": True,
    _finite_keyword: False,
}

_check_array = partial(check_array, **__check_kwargs)


class BaseForest(oneDALEstimator, ABC):
    _onedal_factory = None

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        use_raw_input = get_config().get("use_raw_input", False) is True
        xp, _ = get_namespace(X, y, sample_weight)
        if not use_raw_input:
            X, y = validate_data(
                self,
                X,
                y,
                multi_output=True,
                accept_sparse=False,
                dtype=[xp.float64, xp.float32],
                ensure_all_finite=not sklearn_check_version(
                    "1.4"
                ),  # completed in offload check
                y_numeric=not is_classifier(self),  # trigger for Regressors
            )

            if sample_weight is not None:
                sample_weight = _check_sample_weight(sample_weight, X)

        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if not sklearn_check_version("1.2") and self.criterion == "mse":
            warnings.warn(
                "Criterion 'mse' was deprecated in v1.0 and will be "
                "removed in version 1.2. Use `criterion='squared_error'` "
                "which is equivalent.",
                FutureWarning,
            )

        if y.ndim == 1:
            y = xp.reshape(y, (-1, 1))

        self._n_samples, self.n_outputs_ = y.shape
        self.n_features_in_ = X.shape[1]

        if not use_raw_input:
            if is_classifier(self):
                y, expanded_class_weight = self._validate_y_class_weight_internal(
                    y, sample_weight
                )
            else:
                expanded_class_weight = None

            if expanded_class_weight is not None:
                if sample_weight is not None:
                    sample_weight = sample_weight * expanded_class_weight
                else:
                    sample_weight = expanded_class_weight

            # Decapsulate classes_ attributes following scikit-learn's
            # BaseForest.fit. oneDAL does not support multi-output, therefore
            # the logic can be hardcoded in comparison to scikit-learn's logic
            if hasattr(self, "classes_"):
                self.n_classes_ = self.n_classes_[0]
                self.classes_ = self.classes_[0]

        else:
            # try catch needed for raw_inputs + array_api data where unlike
            # numpy the way to yield unique values is via `unique_values`
            # This should be removed when refactored for gpu zero-copy
            if is_classifier(self):
                try:
                    self.classes_ = xp.unique(y)
                except AttributeError:
                    self.classes_ = xp.unique_values(y)
                self.n_classes_ = self.classes_.shape[0]

        # conform to scikit-learn internal calculations
        if self.bootstrap:
            if sklearn_check_version("1.9"):
                self._n_samples_bootstrap = _get_n_samples_bootstrap(
                    n_samples=X.shape[0],
                    max_samples=self.max_samples,
                    sample_weight=sample_weight,
                )
            else:
                self._n_samples_bootstrap = _get_n_samples_bootstrap(
                    n_samples=X.shape[0], max_samples=self.max_samples
                )
        else:
            self._n_samples_bootstrap = None

        if (self.random_state is not None) and (not daal_check_version((2024, "P", 0))):
            warnings.warn(
                "Setting 'random_state' value is not supported. "
                "State set by oneDAL to default value (777).",
                RuntimeWarning,
            )

        rs = check_random_state(self.random_state)
        # use numpy here due to lack of array API support in sklearn random state
        # seed is a python integer
        seed = rs.randint(0, np.iinfo("i").max)

        # These parameters need to reference onedal.ensemble._forest, as some parameters
        # use defaults set in that module
        onedal_params = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self._to_absolute_max_features(
                self.max_features, self.n_features_in_
            ),
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
            "min_impurity_split": None,
            "bootstrap": self.bootstrap,
            "random_state": seed,
            "observations_per_tree_fraction": (
                self._n_samples_bootstrap / self._n_samples
                if self._n_samples_bootstrap is not None
                else 1.0
            ),
            "max_bins": self.max_bins,
            "min_bin_size": self.min_bin_size,
            # voting mode is set by onedal estimator defaults
            "error_metric_mode": self._err if self.oob_score else "none",
            "variable_importance_mode": "mdi",
            # algorithm mode is set by onedal estimator defaults
        }

        # Lazy evaluation of estimators_
        self._cached_estimators_ = None

        # Compute
        self._onedal_estimator = self._onedal_factory(**onedal_params)
        # class count setting taken via a getattr, which n_classes_ only defined for
        # Classifiers
        self._onedal_estimator.fit(
            X,
            xp.reshape(y, (-1,)),
            sample_weight,
            getattr(self, "n_classes_", 0),
            queue=queue,
        )

        self._save_attributes(xp)

        return self

    def _save_attributes(self, xp):
        if self.oob_score:
            self.oob_score_ = self._onedal_estimator.oob_error_

        self._validate_estimator()

    def _onedal_fit_ready(self, patching_status, X, y, sample_weight):

        patching_status.and_conditions(
            [
                (
                    self.oob_score
                    and daal_check_version((2021, "P", 500))
                    or not self.oob_score,
                    "OOB score is only supported starting from 2021.5 version of oneDAL.",
                ),
                (self.warm_start is False, "Warm start is not supported."),
                (
                    self.ccp_alpha == 0.0,
                    f"Non-zero 'ccp_alpha' ({self.ccp_alpha}) is not supported.",
                ),
                (
                    not is_sparse(X) and not is_sparse(y),
                    "Sparse inputs are not supported.",
                ),
                (
                    self.n_estimators <= 6024,
                    "More than 6024 estimators is not supported.",
                ),
                # Note: multi-valued 'class_weight' is only applicable to multi-output 'y',
                # which is not supported by oneDAL either way.
                (
                    not (
                        hasattr(self, "class_weight")
                        and self.class_weight is not None
                        and not isinstance(self.class_weight, (str, dict))
                    ),
                    "Multi-valued class_weight is not supported",
                ),
                (
                    not self.bootstrap or self.class_weight != "balanced_subsample",
                    "'balanced_subsample' for class_weight is not supported",
                ),
            ]
        )

        if patching_status.get_status() and sklearn_check_version("1.4"):
            try:
                X_test = _check_array(X)
                assert_all_finite(X_test)  # minimally verify the data
                input_is_finite = True
            except ValueError:
                input_is_finite = False
            patching_status.and_conditions(
                [
                    (input_is_finite, "Non-finite input is not supported."),
                    (
                        self.monotonic_cst is None,
                        "Monotonicity constraints are not supported.",
                    ),
                ]
            )

        return patching_status

    def _onedal_cpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.ensemble.{class_name}.{method_name}"
        )

        if method_name == "fit":
            patching_status = self._onedal_fit_ready(patching_status, *data)

            patching_status.and_conditions(
                [
                    (
                        daal_check_version((2023, "P", 200))
                        or self.estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.2",
                    )
                ]
            )

        elif method_name in self._n_jobs_supported_onedal_methods:
            X = data[0]

            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained."),
                    (not is_sparse(X), "X is sparse. Sparse input is not supported."),
                    (self.warm_start is False, "Warm start is not supported."),
                    (
                        daal_check_version((2023, "P", 200))
                        or self.estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.2",
                    ),
                    (
                        self.n_outputs_ == 1,
                        f"Number of outputs ({self.n_outputs_}) is not 1.",
                    ),
                ]
            )

            if method_name == "predict_proba":
                patching_status.and_conditions(
                    [
                        (
                            daal_check_version((2021, "P", 400)),
                            "oneDAL version is lower than 2021.4.",
                        )
                    ]
                )

        else:
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        return patching_status

    def _onedal_gpu_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.ensemble.{class_name}.{method_name}"
        )

        if method_name == "fit":
            patching_status = self._onedal_fit_ready(patching_status, *data)

            patching_status.and_conditions(
                [
                    (
                        daal_check_version((2023, "P", 100))
                        or self.estimator.__class__ == DecisionTreeClassifier,
                        "ExtraTrees only supported starting from oneDAL version 2023.1",
                    ),
                    (
                        not self.oob_score,
                        "oob_scores using r2 or accuracy not implemented.",
                    ),
                ]
            )

        elif method_name in self._n_jobs_supported_onedal_methods:
            X = data[0]

            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained"),
                    (
                        not is_sparse(X),
                        "X is sparse. Sparse input is not supported.",
                    ),
                    (self.warm_start is False, "Warm start is not supported."),
                    (
                        daal_check_version((2023, "P", 100)),
                        "ExtraTrees supported starting from oneDAL version 2023.1",
                    ),
                    (
                        self.n_outputs_ == 1,
                        f"Number of outputs ({self.n_outputs_}) is not 1.",
                    ),
                ]
            )

        else:
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        return patching_status

    def _to_absolute_max_features(self, max_features, n_features):
        # This method handles scikit-learn conformance related to the
        # max_features input, and is separated for ease of maintenance.

        if max_features is None:
            return n_features
        if isinstance(max_features, str):
            if max_features == "auto":
                if not sklearn_check_version("1.3"):
                    if sklearn_check_version("1.1"):
                        warnings.warn(
                            "`max_features='auto'` has been deprecated in 1.1 "
                            "and will be removed in 1.3. To keep the past behaviour, "
                            "explicitly set `max_features=1.0` or remove this "
                            "parameter as it is also the default value for "
                            "RandomForestRegressors and ExtraTreesRegressors.",
                            FutureWarning,
                        )
                    return (
                        max(1, int(math.sqrt(n_features)))
                        if isinstance(self, ForestClassifier)
                        else n_features
                    )
            if max_features == "sqrt":
                return max(1, int(math.sqrt(n_features)))
            if max_features == "log2":
                return max(1, int(math.log2(n_features)))
            allowed_string_values = (
                '"sqrt" or "log2"'
                if sklearn_check_version("1.3")
                else '"auto", "sqrt" or "log2"'
            )
            raise ValueError(
                "Invalid value for max_features. Allowed string "
                f"values are {allowed_string_values}."
            )

        if isinstance(max_features, (numbers.Integral, np.integer)):
            return int(max_features)
        if max_features > 0.0:
            return max(1, int(max_features * n_features))
        return 0

    if not sklearn_check_version("1.2"):

        def _check_parameters(self):
            # This provides ensemble parameter checks for older versions
            # which were centralized in sklearn 1.2. Needed for sklearn
            # conformance.
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
                        (
                            "max_leaf_nodes {0} must be either None " "or larger than 1"
                        ).format(self.max_leaf_nodes)
                    )
            if isinstance(self.max_bins, numbers.Integral):
                if not 2 <= self.max_bins:
                    raise ValueError(
                        "max_bins must be at least 2, got %s" % self.max_bins
                    )
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
                    "min_bin_size must be integral number but was "
                    "%r" % self.min_bin_size
                )

    @property
    def estimators_(self):
        if hasattr(self, "_cached_estimators_"):
            if self._cached_estimators_ is None:
                self._estimators_()
            return self._cached_estimators_
        else:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute 'estimators_'"
            )

    @estimators_.setter
    def estimators_(self, estimators):
        # Needed to allow for proper sklearn operation in fallback mode
        self._cached_estimators_ = estimators

    def _estimators_(self):
        """This attribute provides lazy creation of scikit-learn conformant
        Decision Trees used for analysis in methods such as 'apply'. This will stay
        array_api non-conformant as this is inherently creating sklearn
        objects which are not array_api conformant"""
        # _estimators_ should only be called if _onedal_estimator exists
        check_is_fitted(self, "_onedal_estimator")
        if hasattr(self, "n_classes_"):
            n_classes_ = (
                self.n_classes_
                if isinstance(self.n_classes_, int)
                else self.n_classes_[0]
            )
        else:
            n_classes_ = 1

        # convert model to estimators
        params = {
            "criterion": self.criterion,
            "max_depth": self._onedal_estimator.max_depth,
            "min_samples_split": self._onedal_estimator.min_samples_split,
            "min_samples_leaf": self._onedal_estimator.min_samples_leaf,
            "min_weight_fraction_leaf": self._onedal_estimator.min_weight_fraction_leaf,
            "max_features": self._onedal_estimator.max_features,
            "max_leaf_nodes": self._onedal_estimator.max_leaf_nodes,
            "min_impurity_decrease": self._onedal_estimator.min_impurity_decrease,
            "random_state": None,
        }
        est = self.estimator.__class__(**params)
        # we need to set est.tree_ field with Trees constructed from
        # oneAPI Data Analytics Library solution
        estimators_ = []

        random_state_checked = check_random_state(self.random_state)

        for i in range(self._onedal_estimator.n_estimators):
            est_i = clone(est)
            est_i.set_params(
                random_state=random_state_checked.randint(np.iinfo(np.int32).max)
            )
            est_i.n_features_in_ = self.n_features_in_
            est_i.n_outputs_ = self.n_outputs_
            est_i.n_classes_ = n_classes_
            tree_i_state_class = self._get_tree_state(
                self._onedal_estimator._onedal_model, i, n_classes_
            )
            tree_i_state_dict = {
                "max_depth": tree_i_state_class.max_depth,
                "node_count": tree_i_state_class.node_count,
                "nodes": check_tree_nodes(tree_i_state_class.node_ar),
                "values": tree_i_state_class.value_ar,
            }
            # Note: only on host.
            est_i.tree_ = Tree(
                self.n_features_in_,
                np.array([n_classes_], dtype=np.intp),
                self.n_outputs_,
            )
            est_i.tree_.__setstate__(tree_i_state_dict)
            estimators_.append(est_i)

        self._cached_estimators_ = estimators_

    if not sklearn_check_version("1.2"):

        @property
        def base_estimator(self):
            return self.estimator

        @base_estimator.setter
        def base_estimator(self, estimator):
            self.estimator = estimator


class ForestClassifier(BaseForest, _sklearn_ForestClassifier):
    # Surprisingly, even though scikit-learn warns against using
    # their ForestClassifier directly, it actually has a more stable
    # API than the user-facing objects (over time). If they change it
    # significantly at some point then this may need to be versioned.

    _err = "out_of_bag_error_accuracy|out_of_bag_error_decision_function"
    _get_tree_state = staticmethod(get_tree_state_cls)

    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        max_samples=None,
    ):
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            max_samples=max_samples,
        )

        # The estimator is checked against the class attribute for conformance.
        # This should only trigger if the user uses this class directly.
        if self.estimator.__class__ == DecisionTreeClassifier and not issubclass(
            self._onedal_factory, onedal_RandomForestClassifier
        ):
            self._onedal_factory = onedal_RandomForestClassifier
        elif self.estimator.__class__ == ExtraTreeClassifier and not issubclass(
            self._onedal_factory, onedal_ExtraTreesClassifier
        ):
            self._onedal_factory = onedal_ExtraTreesClassifier

        if self._onedal_factory is None:
            raise TypeError(f" oneDAL estimator has not been set.")

    decision_path = support_input_format(_sklearn_ForestClassifier.decision_path)
    apply = support_input_format(_sklearn_ForestClassifier.apply)

    def _estimators_(self):
        super()._estimators_()
        for est in self._cached_estimators_:
            est.classes_ = self.classes_

    if sklearn_check_version("1.9"):

        def _validate_y_class_weight(self, y, sample_weight):
            return self._validate_y_class_weight_internal(y, sample_weight)

    else:

        def _validate_y_class_weight(self, y):
            return self._validate_y_class_weight_internal(y, None)

    def _validate_y_class_weight_internal(self, y, sample_weight=None):

        xp, is_array_api_compliant = get_namespace(y, sample_weight)

        if not is_array_api_compliant:
            if sklearn_check_version("1.9"):
                return super()._validate_y_class_weight(y, sample_weight)
            else:
                return super()._validate_y_class_weight(y)

        # array API-only branch. This is meant only for the sklearnex
        # forest estimators which assume n_outputs = 1, will not interact
        # with `warm_start`, `balanced_subsample` is not supported and has
        # `class_weight parameter validation elsewhere (which occurs before
        # array API support began in sklearn).

        # only works with 1d array API inputs due to indexing issues
        y = xp.reshape(y, (-1,))
        check_classification_targets(y)

        expanded_class_weight = None

        classes, y_store_unique_indices = xp.unique_inverse(y)

        # force 2d to match sklearn return
        self.classes_ = [classes]
        self.n_classes_ = [classes.shape[0]]

        if self.class_weight is not None:
            class_weights = _compute_class_weight(
                self.class_weight, classes=classes, y=y_store_unique_indices
            )
            expanded_class_weight = xp.ones_like(y)
            # This for loop is O(n*m) where n is # of classes and m # of samples
            # sklearn's compute_sample_weight (roughly equivalent function) uses
            # np.searchsorted which is roughly O((log(n)*m) but unavailable in
            # the array API standard. Be wary of large class counts.
            for i, v in enumerate(class_weights):
                expanded_class_weight[y_store_unique_indices == i] *= v

            # force 2d to match sklearn
            expanded_class_weight = xp.reshape(expanded_class_weight, (-1, 1))
        y = xp.reshape(y_store_unique_indices, (-1, 1))

        return y, expanded_class_weight

    def _save_attributes(self, xp):
        # This assumes that the error_metric_mode variable is set to ._err
        # class attribute
        if self.oob_score:
            # dimension changes and conversion to python types required by sklearn
            # conformance, it is known to be 2d from oneDAL tables
            self.oob_score_ = float(self._onedal_estimator.oob_err_accuracy_[0, 0])
            self.oob_decision_function_ = (
                self._onedal_estimator.oob_err_decision_function_
            )
            if xp.any(self.oob_decision_function_ == 0):
                warnings.warn(
                    "Some inputs do not have OOB scores. This probably means "
                    "too few trees were used to compute any reliable OOB "
                    "estimates.",
                    UserWarning,
                )

        self._validate_estimator()

    def _onedal_fit_ready(self, patching_status, X, y, sample_weight):

        patching_status = super()._onedal_fit_ready(patching_status, X, y, sample_weight)

        if patching_status.get_status():
            xp, is_array_api_compliant = get_namespace(X, y, sample_weight)

            try:
                # properly verifies all non array API inputs without conversion
                correct_target = type_of_target(y) in ["binary", "multiclass"]
            except IndexError:
                # handle array API issues where type_of_target for 2D data
                # which is not supported in type_of_target.
                correct_target = _num_features(y, fallback_1d=True) == 1

            patching_status.and_conditions(
                [
                    (
                        self.criterion == "gini",
                        f"'{self.criterion}' criterion is not supported. "
                        "Only 'gini' criterion is supported.",
                    ),
                    (correct_target, "Only single output classification data supported"),
                    (
                        (
                            xp.unique_values(y)
                            if is_array_api_compliant
                            else xp.unique(xp.asarray(y))
                        ).shape[0]
                        > 1,
                        "Number of classes must be at least 2.",
                    ),
                ]
            )

        return patching_status

    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        else:
            self._check_parameters()

        # slight variation on scikit-learn-intelex rules. This is a custom
        # parameter check which is not covered by self._validate_params()
        # but is necessary for correct math and scikit-learn conformance.
        if not self.bootstrap:
            if self.oob_score:
                raise ValueError("Out of bag estimation only available if bootstrap=True")
            elif self.max_samples is not None:
                raise ValueError(
                    "`max_sample` cannot be set if `bootstrap=False`. "
                    "Either switch to `bootstrap=True` or set "
                    "`max_sample=None`."
                )

        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_ForestClassifier.fit,
            },
            X,
            y,
            sample_weight,
        )
        return self

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_ForestClassifier.predict,
            },
            X,
        )

    @wrap_output_data
    def predict_proba(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "predict_proba",
            {
                "onedal": self.__class__._onedal_predict_proba,
                "sklearn": _sklearn_ForestClassifier.predict_proba,
            },
            X,
        )

    def predict_log_proba(self, X):
        xp, _ = get_namespace(X)
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return xp.log(proba)

        else:
            for k in range(self.n_outputs_):
                proba[k] = xp.log(proba[k])

            return proba

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)
        return dispatch(
            self,
            "score",
            {
                "onedal": self.__class__._onedal_score,
                "sklearn": _sklearn_ForestClassifier.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    def _onedal_predict(self, X, queue=None):
        xp, is_array_api_compliant = get_namespace(X, self.classes_)

        if not get_config()["use_raw_input"]:
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                reset=False,
            )

        res = self._onedal_estimator.predict(X, queue=queue)

        if is_array_api_compliant:
            return xp.take(
                xp.asarray(self.classes_, device=getattr(res, "device", None)),
                xp.astype(xp.reshape(res, (-1,)), xp.int64),
            )
        else:
            return xp.take(self.classes_, res.ravel().astype(xp.int64, casting="unsafe"))

    def _onedal_predict_proba(self, X, queue=None):
        xp, _ = get_namespace(X)

        use_raw_input = get_config().get("use_raw_input", False) is True
        if not use_raw_input:
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                reset=False,
            )

        return self._onedal_estimator.predict_proba(X, queue=queue)

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return accuracy_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    fit.__doc__ = _sklearn_ForestClassifier.fit.__doc__
    predict.__doc__ = _sklearn_ForestClassifier.predict.__doc__
    predict_proba.__doc__ = _sklearn_ForestClassifier.predict_proba.__doc__
    predict_log_proba.__doc__ = _sklearn_ForestClassifier.predict_log_proba.__doc__
    score.__doc__ = _sklearn_ForestClassifier.score.__doc__


class ForestRegressor(BaseForest, _sklearn_ForestRegressor):
    _err = "out_of_bag_error_r2|out_of_bag_error_prediction"
    _get_tree_state = staticmethod(get_tree_state_reg)

    def __init__(
        self,
        estimator,
        n_estimators=100,
        *,
        estimator_params=tuple(),
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        super().__init__(
            estimator,
            n_estimators=n_estimators,
            estimator_params=estimator_params,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )

        # The splitter is checked against the class attribute for conformance
        # This should only trigger if the user uses this class directly.
        if self.estimator.__class__ == DecisionTreeRegressor and not issubclass(
            self._onedal_factory, onedal_RandomForestRegressor
        ):
            self._onedal_factory = onedal_RandomForestRegressor
        elif self.estimator.__class__ == ExtraTreeRegressor and not issubclass(
            self._onedal_factory, onedal_ExtraTreesRegressor
        ):
            self._onedal_factory = onedal_ExtraTreesRegressor

        if self._onedal_factory is None:
            raise TypeError(f" oneDAL estimator has not been set.")

    decision_path = support_input_format(_sklearn_ForestRegressor.decision_path)
    apply = support_input_format(_sklearn_ForestRegressor.apply)

    def _save_attributes(self, xp):
        # This assumes that the error_metric_mode variable is set to ._err
        # class attribute
        if self.oob_score:
            # dimension changes and conversion to python types required by sklearn
            # conformance, it is known to be 2d from oneDAL tables
            self.oob_score_ = float(self._onedal_estimator.oob_err_r2_[0, 0])
            self.oob_prediction_ = xp.reshape(
                self._onedal_estimator.oob_err_prediction_, (-1,)
            )
            if xp.any(self.oob_prediction_ == 0):
                warnings.warn(
                    "Some inputs do not have OOB scores. This probably means "
                    "too few trees were used to compute any reliable OOB "
                    "estimates.",
                    UserWarning,
                )

        self._validate_estimator()

    def _onedal_fit_ready(self, patching_status, X, y, sample_weight):

        patching_status = super()._onedal_fit_ready(patching_status, X, y, sample_weight)

        if patching_status.get_status():
            patching_status.and_conditions(
                [
                    (
                        self.criterion in ["mse", "squared_error"],
                        f"'{self.criterion}' criterion is not supported. "
                        "Only 'mse' and 'squared_error' criteria are supported.",
                    ),
                    (
                        _num_features(y, fallback_1d=True) == 1,
                        f"Number of outputs is not 1.",
                    ),
                ]
            )

        return patching_status

    def fit(self, X, y, sample_weight=None):
        if sklearn_check_version("1.2"):
            self._validate_params()
        else:
            self._check_parameters()

        # slight variation on scikit-learn-intelex rules. This is a custom
        # parameter check which is not covered by self._validate_params()
        # but is necessary for correct math and scikit-learn conformance.
        if not self.bootstrap:
            if self.oob_score:
                raise ValueError("Out of bag estimation only available if bootstrap=True")
            elif self.max_samples is not None:
                raise ValueError(
                    "`max_sample` cannot be set if `bootstrap=False`. "
                    "Either switch to `bootstrap=True` or set "
                    "`max_sample=None`."
                )

        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_ForestRegressor.fit,
            },
            X,
            y,
            sample_weight,
        )
        return self

    @wrap_output_data
    def predict(self, X):
        check_is_fitted(self)
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_ForestRegressor.predict,
            },
            X,
        )

    @wrap_output_data
    def score(self, X, y, sample_weight=None):
        check_is_fitted(self)
        return dispatch(
            self,
            "score",
            {
                "onedal": self.__class__._onedal_score,
                "sklearn": _sklearn_ForestRegressor.score,
            },
            X,
            y,
            sample_weight=sample_weight,
        )

    def _onedal_predict(self, X, queue=None):
        check_is_fitted(self, "_onedal_estimator")
        xp, _ = get_namespace(X)
        use_raw_input = get_config().get("use_raw_input", False) is True

        if not use_raw_input:
            X = validate_data(
                self,
                X,
                dtype=[xp.float64, xp.float32],
                reset=False,
            )  # Warning, order of dtype matters

        return self._onedal_estimator.predict(X, queue=queue)

    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        return r2_score(
            y, self._onedal_predict(X, queue=queue), sample_weight=sample_weight
        )

    fit.__doc__ = _sklearn_ForestRegressor.fit.__doc__
    predict.__doc__ = _sklearn_ForestRegressor.predict.__doc__
    score.__doc__ = _sklearn_ForestRegressor.score.__doc__


@enable_array_api("1.6")
@register_hyperparameters({"predict": ("decision_forest", "infer")})
@control_n_jobs(decorated_methods=["fit", "predict", "predict_proba", "score"])
class RandomForestClassifier(ForestClassifier):
    __doc__ = _sklearn_RandomForestClassifier.__doc__
    _onedal_factory = onedal_RandomForestClassifier

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_RandomForestClassifier._parameter_constraints,
            "max_bins": [Interval(numbers.Integral, 2, None, closed="left")],
            "min_bin_size": [Interval(numbers.Integral, 1, None, closed="left")],
        }

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
        ):
            super().__init__(
                DecisionTreeClassifier(),
                n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                    "monotonic_cst",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
            self.monotonic_cst = monotonic_cst

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
        ):
            super().__init__(
                DecisionTreeClassifier(),
                n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size


@enable_array_api("1.5")
@control_n_jobs(decorated_methods=["fit", "predict", "score"])
class RandomForestRegressor(ForestRegressor):
    __doc__ = _sklearn_RandomForestRegressor.__doc__
    _onedal_factory = onedal_RandomForestRegressor

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_RandomForestRegressor._parameter_constraints,
            "max_bins": [Interval(numbers.Integral, 2, None, closed="left")],
            "min_bin_size": [Interval(numbers.Integral, 1, None, closed="left")],
        }

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
        ):
            super().__init__(
                DecisionTreeRegressor(),
                n_estimators=n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                    "monotonic_cst",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
            self.monotonic_cst = monotonic_cst

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
        ):
            super().__init__(
                DecisionTreeRegressor(),
                n_estimators=n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size


@enable_array_api("1.6")
@control_n_jobs(decorated_methods=["fit", "predict", "predict_proba", "score"])
class ExtraTreesClassifier(ForestClassifier):
    __doc__ = _sklearn_ExtraTreesClassifier.__doc__
    _onedal_factory = onedal_ExtraTreesClassifier

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_ExtraTreesClassifier._parameter_constraints,
            "max_bins": [Interval(numbers.Integral, 2, None, closed="left")],
            "min_bin_size": [Interval(numbers.Integral, 1, None, closed="left")],
        }

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
            bootstrap=False,
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
        ):
            super().__init__(
                ExtraTreeClassifier(),
                n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                    "monotonic_cst",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
            self.monotonic_cst = monotonic_cst

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
            bootstrap=False,
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
        ):
            super().__init__(
                ExtraTreeClassifier(),
                n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                class_weight=class_weight,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size


@enable_array_api("1.5")
@control_n_jobs(decorated_methods=["fit", "predict", "score"])
class ExtraTreesRegressor(ForestRegressor):
    __doc__ = _sklearn_ExtraTreesRegressor.__doc__
    _onedal_factory = onedal_ExtraTreesRegressor

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {
            **_sklearn_ExtraTreesRegressor._parameter_constraints,
            "max_bins": [Interval(numbers.Integral, 2, None, closed="left")],
            "min_bin_size": [Interval(numbers.Integral, 1, None, closed="left")],
        }

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
            bootstrap=False,
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
        ):
            super().__init__(
                ExtraTreeRegressor(),
                n_estimators=n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                    "monotonic_cst",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size
            self.monotonic_cst = monotonic_cst

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
            bootstrap=False,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            ccp_alpha=0.0,
            max_samples=None,
            max_bins=256,
            min_bin_size=1,
        ):
            super().__init__(
                ExtraTreeRegressor(),
                n_estimators=n_estimators,
                estimator_params=(
                    "criterion",
                    "max_depth",
                    "min_samples_split",
                    "min_samples_leaf",
                    "min_weight_fraction_leaf",
                    "max_features",
                    "max_leaf_nodes",
                    "min_impurity_decrease",
                    "random_state",
                    "ccp_alpha",
                ),
                bootstrap=bootstrap,
                oob_score=oob_score,
                n_jobs=n_jobs,
                random_state=random_state,
                verbose=verbose,
                warm_start=warm_start,
                max_samples=max_samples,
            )

            self.criterion = criterion
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.min_samples_leaf = min_samples_leaf
            self.min_weight_fraction_leaf = min_weight_fraction_leaf
            self.max_features = max_features
            self.max_leaf_nodes = max_leaf_nodes
            self.min_impurity_decrease = min_impurity_decrease
            self.ccp_alpha = ccp_alpha
            self.max_bins = max_bins
            self.min_bin_size = min_bin_size


# Allow for isinstance calls without inheritance changes using ABCMeta
_sklearn_RandomForestClassifier.register(RandomForestClassifier)
_sklearn_RandomForestRegressor.register(RandomForestRegressor)
_sklearn_ExtraTreesClassifier.register(ExtraTreesClassifier)
_sklearn_ExtraTreesRegressor.register(ExtraTreesRegressor)

# ==============================================================================
# Copyright 2014 Intel Corporation
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

# daal4py DecisionTree scikit-learn-compatible estimator classes

import numbers
import warnings

import numpy as np
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted

import daal4py as d4p

from .._utils import getFPType, make2d


class DecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    Decision tree classifier powered by oneAPI Data Analytics Library.

       https://www.intel.com/content/www/us/en/developer/tools/oneapi/
       onedal-documentation.html

    Parameters
    ----------
    max_depth : int or None, default=None
        Depth of the tree fitten to data. None corresponds to unlimited depth.

    min_observations_in_leaf_node : int, default=10
        The number of estimators in the ensemble.

    estimator_params : list of str, default=tuple()
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.


    Training:
        inputs: dataForPruning, labelsForPruning
        parameters: fptype, method, nClasses, splitCriterion, pruning,
                    maxTreeDepth, minObservationsInLeafNodes

    Prediction:
        parameters: fptype, method, nBins, nClasses,
        resultsToEvaluate (computeClassesLabels|computeClassProbabilities)
        N.B.: The only supported value for current version of the library is nBins=1.
              nBins is the number of bins used to compute probabilities of the
              observations belonging to the class.
    """

    def __init__(
        self, max_depth=None, min_observations_in_leaf_node=1, split_criterion="gini"
    ):
        self.max_depth = max_depth
        self.min_observations_in_leaf_node = min_observations_in_leaf_node
        self.split_criterion = split_criterion

    def _daal4py_fit(self, X, y, w, pruning_set=None):
        X_fptype = getFPType(X)
        X = make2d(X)
        y = make2d(y)

        if pruning_set is None:
            _pruning = "none"
            _pruning_X = None
            _pruning_y = None
        else:
            _pruning = "reducedErrorPruning"
            if isinstance(pruning_set, (tuple, list)) and len(pruning_set) == 2:
                _pruning_X, _pruning_y = pruning_set
                check_consistent_length(_pruning_X, _pruning_y)
                _pruning_X = make2d(_pruning_X)
                _pruning_y = make2d(_pruning_y)
            else:
                raise ValueError(
                    "pruning_set parameter is expected to be "
                    "a tuple of pruning features and pruning "
                    "dependent variables"
                )

        if w is not None:
            w = make2d(np.asarray(w))

        daal_max_tree_depth = 0 if (self.max_depth is None) else int(self.max_depth) + 1
        alg = d4p.decision_tree_classification_training(
            fptype=X_fptype,
            method="defaultDense",
            nClasses=int(self.n_classes_),
            splitCriterion=self.split_criterion,
            maxTreeDepth=daal_max_tree_depth,
            minObservationsInLeafNodes=int(self.min_observations_in_leaf_node),
            pruning=_pruning,
        )
        res = alg.compute(
            X, y, dataForPruning=_pruning_X, labelsForPruning=_pruning_y, weights=w
        )
        self.daal_model_ = res.model
        self._cached_tree_state_ = None

    def _get_tree_state(self):
        """
        Internal utility that returns an array behind scikit-learn's tree object
        from daal_model_ produced by call to fit
        """
        check_is_fitted(self, ["daal_model_", "_cached_tree_state_"])
        if self._cached_tree_state_ is None:
            tree_state_class = d4p.getTreeState(self.daal_model_, int(self.n_classes_))
            self._cached_tree_state_ = tree_state_class
        return self._cached_tree_state_

    def get_n_leaves(self):
        ts = self._get_tree_state()
        return ts.leaf_count

    def get_depth(self):
        ts = self._get_tree_state()
        return ts.max_depth

    def fit(self, X, y, sample_weight=None, pruning_set=None):
        """Build a decision tree classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples. Internally, it will be converted to
            ``dtype=np.float64`` and if a sparse matrix is provided
            to a sparse ``csc_matrix``.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels) as integers or strings.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            that would create child nodes with net zero or negative weight are
            ignored while searching for a split in each node. Splits are also
            ignored if they would result in any single class carrying a
            negative weight in either child node.

        pruning_set: None or a tuple of (X, y) corrsponding to features and
            associated labels used for tree pruning. See [1] for more details.

        Returns
        -------
        self : DecisionTreeClassifier
            Fitted estimator.


        [1] https://www.intel.com/content/www/us/en/developer/tools/oneapi/
        onedal-documentation.html
        """

        if self.split_criterion not in ("gini", "infoGain"):
            raise ValueError(
                'Parameter "split_criterion" must be ' '"gini" or "infoGain".'
            )

        if not isinstance(self.max_depth, numbers.Integral) or self.max_depth < 0:
            if self.max_depth is not None:
                raise ValueError(
                    'Parameter "max_depth" must be '
                    "a non-negative integer value or None."
                )

        if (
            not isinstance(self.min_observations_in_leaf_node, numbers.Integral)
            or self.min_observations_in_leaf_node <= 0
        ):
            raise ValueError(
                'Parameter "min_observations_in_leaf_node" must be '
                "non-zero positive integer value."
            )

        X = check_array(X, dtype=[np.single, np.double])
        y = np.asarray(y)
        y = np.atleast_1d(y)

        if y.ndim == 2 and y.shape[1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        check_consistent_length(X, y)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_outputs_ = y.shape[1]
        if self.n_outputs_ != 1:
            _class_name = self.__class__.__name__
            raise ValueError(
                _class_name + " does not currently support "
                "multi-output data. "
                "Consider using OneHotEncoder"
            )

        y = check_array(y, ensure_2d=False, dtype=None)
        check_classification_targets(y)

        y = np.copy(y)

        self.classes_ = []
        self.n_classes_ = []

        y_store_unique_indices = np.zeros(y.shape, dtype=np.int)
        for k in range(self.n_outputs_):
            classes_k, y_store_unique_indices[:, k] = np.unique(
                y[:, k], return_inverse=True
            )
            self.classes_.append(classes_k)
            self.n_classes_.append(classes_k.shape[0])
        y = y_store_unique_indices

        self.n_classes_ = self.n_classes_[0]
        self.classes_ = self.classes_[0]

        self.n_features_ = X.shape[1]

        if self.n_classes_ < 2:
            raise ValueError("Training data only contain information about one class.")

        self._daal4py_fit(X, y, sample_weight, pruning_set=pruning_set)

        return self

    def _validate_X_predict(self, X, check_input):
        """Validate X whenever one tries to predict, apply, predict_proba"""
        if check_input:
            X = check_array(X, dtype=[np.single, np.double], accept_sparse="csr")
            if issparse(X) and (X.indices.dtype != np.intc or X.indptr.dtype != np.intc):
                raise ValueError("No support for np.int64 index based " "sparse matrices")

        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError(
                "Number of features of the model must "
                "match the input. Model n_features is %s and "
                "input n_features is %s " % (self.n_features_, n_features)
            )

        return X

    def _daal4py_predict(self, X):
        fptype = getFPType(X)
        alg = d4p.decision_tree_classification_prediction(
            fptype=fptype,
            method="defaultDense",
            nBins=1,
            nClasses=self.n_classes_,
            resultsToEvaluate="computeClassLabels",
        )
        res = alg.compute(X, self.daal_model_)
        return res.prediction.ravel()

    def predict(self, X, check_input=True):
        check_is_fitted(self, "daal_model_")
        X = self._validate_X_predict(X, check_input)
        y = self._daal4py_predict(X)
        return self.classes_.take(np.asarray(y, dtype=np.intp), axis=0)

    def predict_proba(self, X, check_input=True):
        check_is_fitted(self, "daal_model_")
        X = self._validate_X_predict(X, check_input)
        y = self._daal4py_predict(X)
        return self.classes_.take(np.asarray(y, dtype=np.intp), axis=0)

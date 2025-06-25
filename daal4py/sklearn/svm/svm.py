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

from __future__ import print_function

import numpy as np
import sklearn.svm._base as svm_base
import sklearn.svm._classes as svm_classes
from scipy import sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import _ovr_decision_function
from sklearn.utils.validation import (
    _check_sample_weight,
    _num_samples,
    check_consistent_length,
    check_is_fitted,
)

import daal4py

from .._utils import PatchingConditionsChain, getFPType, make2d
from ..utils.validation import validate_data


def _get_libsvm_impl():
    return ["c_svc", "nu_svc", "one_class", "epsilon_svr", "nu_svr"]


def _dual_coef_getter(self):
    return self._internal_dual_coef_


def _intercept_getter(self):
    return self._internal_intercept_


def _dual_coef_setter(self, val):
    self._internal_dual_coef_ = val
    if hasattr(self, "daal_model_"):
        del self.daal_model_
    if getattr(self, "_daal_fit", False):
        self._daal_fit = False


def _intercept_setter(self, val):
    self._internal_intercept_ = val
    if hasattr(self, "daal_model_"):
        del self.daal_model_
    if getattr(self, "_daal_fit", False):
        self._daal_fit = False


# Methods to extract coefficients
def group_indices_by_class(num_classes, sv_ind_by_clf, labels):
    sv_ind_counters = np.zeros(num_classes, dtype=np.intp)

    num_of_sv_per_class = np.bincount(labels[np.hstack(sv_ind_by_clf)])
    sv_ind_by_class = [np.empty(n, dtype=np.int32) for n in num_of_sv_per_class]

    for indices_per_clf in sv_ind_by_clf:
        for sv_index in indices_per_clf:
            sv_label = labels[sv_index]
            i = sv_ind_counters[sv_label]
            sv_ind_by_class[sv_label][i] = sv_index
            sv_ind_counters[sv_label] += 1

    return sv_ind_by_class


def map_sv_to_columns_in_dual_coef_matrix(sv_ind_by_class):
    from collections import defaultdict

    sv_ind_mapping = defaultdict(lambda: -1)
    p = 0
    for indices_per_class in sv_ind_by_class:
        indices_per_class.sort()
        for sv_index in indices_per_class:
            if sv_ind_mapping[sv_index] == -1:
                sv_ind_mapping[sv_index] = p
                p += 1
    return sv_ind_mapping


def map_to_lexicographic(n):
    """Returns permutation of reverse lexicographics to
    lexicographics orders for pairs of n consecutive integer indexes
    """
    from itertools import combinations, count

    two_class_order_gen = ((j, i) for i in range(n) for j in range(i))
    reverse_lookup = {key: val for key, val in zip(two_class_order_gen, count(0))}
    perm_iter = (reverse_lookup[pair] for pair in combinations(range(n), 2))
    return np.fromiter(perm_iter, dtype=np.intp)


def permute_list(li, perm):
    "Rearrange `li` according to `perm`"
    return [li[i] for i in perm]


def extract_dual_coef(num_classes, sv_ind_by_clf, sv_coef_by_clf, labels):
    """Construct dual coefficients array in SKLearn peculiar layout,
    as well corresponding support vector indexes
    """
    sv_ind_by_class = group_indices_by_class(num_classes, sv_ind_by_clf, labels)
    sv_ind_mapping = map_sv_to_columns_in_dual_coef_matrix(sv_ind_by_class)

    num_unique_sv = len(sv_ind_mapping)
    dc_dt = sv_coef_by_clf[0].dtype

    dual_coef = np.zeros((num_classes - 1, num_unique_sv), dtype=dc_dt)
    support_ = np.empty((num_unique_sv,), dtype=np.int32)

    p = 0
    for i in range(0, num_classes):
        for j in range(i + 1, num_classes):
            sv_ind_i_vs_j = sv_ind_by_clf[p]
            sv_coef_i_vs_j = sv_coef_by_clf[p]
            p += 1

            for k, sv_index in enumerate(sv_ind_i_vs_j):
                label = labels[sv_index]
                col_index = sv_ind_mapping[sv_index]
                if j == label:
                    row_index = i
                else:
                    row_index = j - 1
                dual_coef[row_index, col_index] = sv_coef_i_vs_j[k]
                support_[col_index] = sv_index

    return dual_coef, support_


def _daal4py_kf(kernel, X_fptype, gamma=1.0, is_sparse=False):
    method = "fastCSR" if is_sparse else "defaultDense"
    if kernel == "rbf":
        sigma_value = np.sqrt(0.5 / gamma)
        kf = daal4py.kernel_function_rbf(
            fptype=X_fptype, method=method, sigma=sigma_value
        )
    elif kernel == "linear":
        kf = daal4py.kernel_function_linear(fptype=X_fptype, method=method)
    else:
        raise ValueError(
            "_daal4py_fit received unexpected kernel specification {}.".format(kernel)
        )

    return kf


def _daal4py_check_weight(self, X, y, sample_weight):
    ww = None
    if sample_weight.shape[0] > 0:
        sample_weight = _check_sample_weight(sample_weight, X)
        if np.all(sample_weight <= 0):
            raise ValueError("Invalid input - all samples have zero or negative weights.")
        if np.any(sample_weight <= 0):
            if len(np.unique(y[sample_weight > 0])) != len(self.classes_):
                raise ValueError(
                    "Invalid input - all samples with positive weights "
                    "have the same label."
                )
        ww = sample_weight
    elif self.class_weight is not None:
        ww = np.ones(X.shape[0], dtype=np.float64)
    if self.class_weight is not None:
        for i, v in enumerate(self.class_weight_):
            ww[y == i] *= v
    return ww


def _daal4py_svm(
    fptype,
    C,
    accuracyThreshold,
    tau,
    maxIterations,
    cacheSize,
    doShrinking,
    kernel,
    nClasses=2,
):
    svm_train = daal4py.svm_training(
        method="thunder",
        fptype=fptype,
        C=C,
        accuracyThreshold=accuracyThreshold,
        tau=tau,
        maxIterations=maxIterations,
        cacheSize=cacheSize,
        doShrinking=doShrinking,
        kernel=kernel,
    )
    if nClasses == 2:
        algo = svm_train
    else:
        algo = daal4py.multi_class_classifier_training(
            nClasses=nClasses,
            fptype=fptype,
            method="oneAgainstOne",
            training=svm_train,
        )

    return algo


def _daal4py_fit(self, X, y_inp, sample_weight, kernel, is_sparse=False):
    if self.C <= 0:
        raise ValueError("C <= 0")
    num_classes = len(self.classes_)

    if sample_weight is not None:
        sample_weight = make2d(sample_weight)

    y = make2d(y_inp)
    X_fptype = getFPType(X)
    kf = _daal4py_kf(kernel, X_fptype, gamma=self._gamma, is_sparse=is_sparse)
    algo = _daal4py_svm(
        fptype=X_fptype,
        C=float(self.C),
        accuracyThreshold=float(self.tol),
        tau=1e-12,
        maxIterations=int(self.max_iter if self.max_iter > 0 else 2**30),
        cacheSize=int(self.cache_size * 1024 * 1024),
        doShrinking=bool(self.shrinking),
        kernel=kf,
        nClasses=num_classes,
    )

    res = algo.compute(data=X, labels=y, weights=sample_weight)

    model = res.model
    self.daal_model_ = model

    if num_classes == 2:
        # binary
        two_class_sv_ind_ = model.SupportIndices
        two_class_sv_ind_ = two_class_sv_ind_.ravel()

        # support indexes need permutation to arrange them
        # into the same layout as that of Scikit-Learn
        tmp = np.empty(
            two_class_sv_ind_.shape,
            dtype=np.dtype([("label", y.dtype), ("ind", two_class_sv_ind_.dtype)]),
        )
        tmp["label"][:] = y[two_class_sv_ind_].ravel()
        tmp["ind"][:] = two_class_sv_ind_
        perm = np.argsort(tmp, order=["label", "ind"])
        del tmp

        self.support_ = two_class_sv_ind_[perm]
        self.support_vectors_ = X[self.support_]

        self.dual_coef_ = model.ClassificationCoefficients.T
        if is_sparse:
            self.dual_coef_ = sp.csr_matrix(self.dual_coef_)
        self.dual_coef_ = self.dual_coef_[:, perm]
        self.intercept_ = np.array([model.Bias])

    else:
        # multi-class
        intercepts = []
        coefs = []
        sv_ind_by_clf = []
        label_indexes = []

        model_id = 0
        for i1 in range(num_classes):
            label_indexes.append(np.where(y == i1)[0])
            for i2 in range(i1):
                svm_model = model.TwoClassClassifierModel(model_id)

                # Indices correspond to input features with label i1
                # followed by input features with label i2
                two_class_sv_ind_ = svm_model.SupportIndices
                # Map these indexes to indexes of the training data
                sv_ind = np.take(
                    np.hstack((label_indexes[i1], label_indexes[i2])),
                    two_class_sv_ind_.ravel(),
                )
                sv_ind_by_clf.append(sv_ind)

                # svs_ = getArrayFromNumericTable(svm_model.getSupportVectors())
                # assert np.array_equal(svs_, X[sv_ind])

                intercepts.append(-svm_model.Bias)
                coefs.append(-svm_model.ClassificationCoefficients)
                model_id += 1

        # permute solutions to lexicographic ordering
        to_lex_perm = map_to_lexicographic(num_classes)
        sv_ind_by_clf = permute_list(sv_ind_by_clf, to_lex_perm)
        sv_coef_by_clf = permute_list(coefs, to_lex_perm)
        intercepts = permute_list(intercepts, to_lex_perm)

        self.dual_coef_, self.support_ = extract_dual_coef(
            num_classes,  # number of classes
            sv_ind_by_clf,  # support vector indexes by two-class classifiers
            sv_coef_by_clf,  # classification coefficients by two-class classifiers
            y.squeeze().astype(np.intp, copy=False),  # integer labels
        )
        if is_sparse:
            self.dual_coef_ = sp.csr_matrix(self.dual_coef_)
        self.support_vectors_ = X[self.support_]
        self.intercept_ = np.array(intercepts)

    indices = y.take(self.support_, axis=0)
    self._n_support = np.array(
        [np.sum(indices == i) for i, c in enumerate(self.classes_)], dtype=np.int32
    )

    self._probA = np.empty(0)
    self._probB = np.empty(0)


def _compute_gamma(gamma, kernel, X):
    """
    Computes actual value of 'gamma' parameter of RBF kernel
    corresponding to SVC keyword values `gamma` and `kernel`, and feature
    matrix X, with sparsity `sparse`.

    In 0.20 gamma='scale' used to mean compute 'gamma' based on
    column-wise standard deviation, but in 0.20.3 it was changed
    to use column-wise variance.

    See: https://github.com/scikit-learn/scikit-learn/pull/13221
    """
    if gamma == "scale":
        kernel_uses_gamma = not callable(kernel) and kernel not in (
            "linear",
            "precomputed",
        )
        if kernel_uses_gamma:
            if sp.isspmatrix(X):
                # var = E[X^2] - E[X]^2
                X_sc = (X.multiply(X)).mean() - (X.mean()) ** 2
            else:
                X_sc = X.var()
        else:
            X_sc = 1.0 / X.shape[1]
        if gamma == "scale":
            if X_sc != 0:
                _gamma = 1.0 / (X.shape[1] * X_sc)
            else:
                _gamma = 1.0
        else:
            _gamma = 1.0 / X.shape[1]
    elif gamma == "auto":
        _gamma = 1.0 / X.shape[1]
    elif isinstance(gamma, str):
        raise ValueError(
            "When 'gamma' is a string, it should be either 'scale' or "
            "'auto'. Got '{}' instead.".format(gamma)
        )
    else:
        _gamma = gamma

    return _gamma


def fit(self, X, y, sample_weight=None):
    """Fit the SVM model according to the given training data.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training vectors, where n_samples is the number of samples
        and n_features is the number of features.
        For kernel="precomputed", the expected shape of X is
        (n_samples, n_samples).

    y : array-like, shape (n_samples,)
        Target values (class labels in classification, real numbers in
        regression)

    sample_weight : array-like, shape (n_samples,)
        Per-sample weights. Rescale C per sample. Higher weights
        force the classifier to put more emphasis on these points.

    Returns
    -------
    self : object

    Notes
    ------
    If X and y are not C-ordered and contiguous arrays of np.float64 and
    X is not a scipy.sparse.csr_matrix, X and/or y may be copied.

    If X is a dense array, then the other methods will not support sparse
    matrices as input.
    """
    rnd = check_random_state(self.random_state)

    is_sparse = sp.isspmatrix(X)
    if is_sparse and self.kernel == "precomputed":
        raise TypeError("Sparse precomputed kernels are not supported.")
    self._sparse = is_sparse and not callable(self.kernel)

    if hasattr(self, "decision_function_shape"):
        if self.decision_function_shape not in ("ovr", "ovo"):
            raise ValueError(
                f"decision_function_shape must be either 'ovr' or 'ovo', "
                f"got {self.decision_function_shape}."
            )

    if callable(self.kernel):
        check_consistent_length(X, y)
    else:
        X, y = validate_data(
            self,
            X,
            y,
            dtype=np.float64,
            order="C",
            accept_sparse="csr",
            accept_large_sparse=False,
        )
    y = self._validate_targets(y)

    sample_weight = np.asarray(
        [] if sample_weight is None else sample_weight, dtype=np.float64
    )
    solver_type = _get_libsvm_impl().index(self._impl)

    # input validation
    n_samples = _num_samples(X)
    if solver_type != 2 and n_samples != y.shape[0]:
        raise ValueError(
            "X and y have incompatible shapes.\n"
            "X has %s samples, but y has %s." % (n_samples, y.shape[0])
        )

    if self.kernel == "precomputed" and n_samples != X.shape[1]:
        raise ValueError("X.shape[0] should be equal to X.shape[1]")

    if sample_weight.shape[0] > 0 and sample_weight.shape[0] != n_samples:
        raise ValueError(
            "sample_weight and X have incompatible shapes: "
            "%r vs %r\n"
            "Note: Sparse matrices cannot be indexed w/"
            "boolean masks (use `indices=True` in CV)." % (sample_weight.shape, X.shape)
        )

    kernel = "precomputed" if callable(self.kernel) else self.kernel
    if kernel == "precomputed":
        self._gamma = 0.0
    else:
        self._gamma = _compute_gamma(self.gamma, kernel, X)

    fit = self._sparse_fit if self._sparse else self._dense_fit
    if self.verbose:  # pragma: no cover
        print("[LibSVM]", end="")

    # see comment on the other call to np.iinfo in this file
    seed = rnd.randint(np.iinfo("i").max)

    _patching_status = PatchingConditionsChain("sklearn.svm.SVC.fit")
    _dal_ready = _patching_status.and_conditions(
        [
            (
                kernel in ["linear", "rbf"],
                f"'{kernel}' kernel is not supported. "
                "Only 'linear' and 'rbf' kernels are supported.",
            )
        ]
    )
    _patching_status.write_log()
    if _dal_ready:
        sample_weight = _daal4py_check_weight(self, X, y, sample_weight)

        self._daal_fit = True
        _daal4py_fit(self, X, y, sample_weight, kernel, is_sparse=is_sparse)
        self.fit_status_ = 0

        if self.probability:
            params = self.get_params()
            params["probability"] = False
            params["decision_function_shape"] = "ovr"
            clf_base = SVC(**params)
            try:
                n_splits = 5
                cv = StratifiedKFold(
                    n_splits=n_splits, shuffle=True, random_state=self.random_state
                )
                self.clf_prob = CalibratedClassifierCV(
                    clf_base, ensemble=False, cv=cv, method="sigmoid", n_jobs=n_splits
                )

                self.clf_prob.fit(X, y, sample_weight)
            except ValueError:
                clf_base = clf_base.fit(X, y, sample_weight)
                self.clf_prob = CalibratedClassifierCV(
                    clf_base, cv="prefit", method="sigmoid"
                )
                self.clf_prob.fit(X, y, sample_weight)
    else:
        self._daal_fit = False
        fit(X, y, sample_weight, solver_type, kernel, random_seed=seed)

    self.shape_fit_ = X.shape if hasattr(X, "shape") else (n_samples,)

    # In binary case, we need to flip the sign of coef, intercept and
    # decision function. Use self._intercept_ and self._dual_coef_ internally.
    if not getattr(self, "_daal_fit", False):
        self._internal_intercept_ = self.intercept_.copy()
        self._internal_dual_coef_ = self.dual_coef_.copy()
    else:
        self._internal_intercept_ = self.intercept_.copy()
        self._internal_dual_coef_ = self.dual_coef_.copy()
        if len(self.classes_) == 2:
            self._internal_dual_coef_ *= -1
            self._internal_intercept_ *= -1

    if (
        not getattr(self, "_daal_fit", False)
        and len(self.classes_) == 2
        and self._impl in ["c_svc", "nu_svc"]
    ):
        self.intercept_ *= -1
        self.dual_coef_ *= -1

    return self


def _daal4py_predict(self, X, is_decision_function=False):
    X_fptype = getFPType(X)
    num_classes = len(self.classes_)

    kf = _daal4py_kf(self.kernel, X_fptype, gamma=self._gamma, is_sparse=sp.isspmatrix(X))

    svm_predict = daal4py.svm_prediction(
        fptype=X_fptype, method="defaultDense", kernel=kf
    )
    if num_classes == 2:
        alg = svm_predict
    else:
        result_to_compute = (
            "computeDecisionFunction" if is_decision_function else "computeClassLabels"
        )
        alg = daal4py.multi_class_classifier_prediction(
            nClasses=num_classes,
            fptype=X_fptype,
            pmethod="voteBased",
            tmethod="oneAgainstOne",
            resultsToEvaluate=result_to_compute,
            prediction=svm_predict,
        )

    predictionRes = alg.compute(X, self.daal_model_)
    if not is_decision_function or num_classes == 2:
        res = predictionRes.prediction
        res = res.ravel()
    else:
        res = -predictionRes.decisionFunction

    if num_classes == 2 and not is_decision_function:
        # Convert from oneAPI Data Analytics Library format back to
        # original classes
        np.greater(res, 0, out=res)

    return res


def predict(self, X):
    check_is_fitted(self)

    _break_ties = getattr(self, "break_ties", False)
    if _break_ties and self.decision_function_shape == "ovo":
        raise ValueError(
            "break_ties must be False when " "decision_function_shape is 'ovo'"
        )

    _patching_status = PatchingConditionsChain("sklearn.svm.SVC.predict")
    _dal_ready = _patching_status.and_conditions(
        [
            (not _break_ties, "Breaking ties is not supported."),
            (
                self.decision_function_shape != "ovr",
                "'ovr' decision function shape is not supported.",
            ),
            (len(self.classes_) <= 2, "Number of classes > 2."),
        ],
        conditions_merging=any,
    )
    _patching_status.write_log()
    if not _dal_ready:
        y = np.argmax(self.decision_function(X), axis=1)
    else:
        X = self._validate_for_predict(X)
        _dal_ready = _patching_status.and_conditions(
            [
                (
                    getattr(self, "_daal_fit", False) and hasattr(self, "daal_model_"),
                    "oneDAL model was not trained.",
                )
            ]
        )
        if _dal_ready:
            if self.probability and self.clf_prob is not None:
                y = self.clf_prob.predict(X)
            else:
                y = _daal4py_predict(self, X)
        else:
            predict_func = self._sparse_predict if self._sparse else self._dense_predict
            y = predict_func(X)

    return self.classes_.take(np.asarray(y, dtype=np.intp))


def _daal4py_predict_proba(self, X):
    X = self._validate_for_predict(X)

    if getattr(self, "clf_prob", None) is None:
        raise NotFittedError(
            "predict_proba is not available when fitted with probability=False"
        )
    prob = self.clf_prob.predict_proba(X)
    return prob


@property
def predict_proba(self):
    self._check_proba()
    _patching_status = PatchingConditionsChain("sklearn.svm.SVC.predict_proba")
    _dal_ready = _patching_status.and_conditions(
        [(getattr(self, "_daal_fit", False), "oneDAL model was not trained.")]
    )
    _patching_status.write_log()
    if _dal_ready:
        algo = self._daal4py_predict_proba
    else:
        algo = self._predict_proba
    return algo


def decision_function(self, X):
    _patching_status = PatchingConditionsChain("sklearn.svm.SVC.decision_function")
    _dal_ready = _patching_status.and_conditions(
        [(getattr(self, "_daal_fit", False), "oneDAL model was not trained.")]
    )
    _patching_status.write_log()
    if _dal_ready:
        X = self._validate_for_predict(X)
        dec = _daal4py_predict(self, X, is_decision_function=True)
    else:
        dec = self._decision_function(X)
    if self.decision_function_shape == "ovr" and len(self.classes_) > 2:
        return _ovr_decision_function(dec < 0, -dec, len(self.classes_))
    return dec


__base_svc_init_arg_names__ = []

__base_svc_init_function__ = svm_base.BaseSVC.__init__
__base_svc_init_function_code__ = __base_svc_init_function__.__code__

try:
    # retrieve tuple of code argument names to check whether
    # new in 0.22 keyword 'break_ties' is in it
    __base_svc_init_arg_names__ = __base_svc_init_function_code__.co_varnames
except AttributeError:
    pass

del __base_svc_init_function__
del __base_svc_init_function_code__


class SVC(svm_base.BaseSVC):
    _impl = "c_svc"

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
    ):
        super(SVC, self).__init__(
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            C=C,
            nu=0.0,
            shrinking=shrinking,
            probability=probability,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )


SVC.fit = fit
SVC.predict = predict
SVC.predict_proba = predict_proba
SVC.decision_function = decision_function
SVC._daal4py_predict_proba = _daal4py_predict_proba
SVC._dual_coef_ = property(_dual_coef_getter, _dual_coef_setter)
SVC._intercept_ = property(_intercept_getter, _intercept_setter)
SVC.__doc__ = svm_classes.SVC.__doc__
SVC.fit.__doc__ = fit.__doc__
SVC.predict.__doc__ = predict.__doc__
SVC.predict_proba.__doc__ = predict_proba.__doc__
SVC.decision_function.__doc__ = decision_function.__doc__

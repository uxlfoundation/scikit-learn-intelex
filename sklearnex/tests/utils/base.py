# ==============================================================================
# Copyright 2024 Intel Corporation
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

import platform
import subprocess
from functools import partial
from inspect import Parameter, getattr_static, isclass, signature

import numpy as np
from scipy import sparse as sp
from sklearn import clone
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    ClusterMixin,
    OutlierMixin,
    RegressorMixin,
    TransformerMixin,
)
from sklearn.datasets import load_diabetes, load_iris
from sklearn.neighbors._base import KNeighborsMixin
from sklearn.utils.validation import check_is_fitted

from onedal.datatypes import from_table, to_table
from onedal.tests.utils._dataframes_support import _convert_to_dataframe
from onedal.utils._array_api import _get_sycl_namespace
from sklearnex import get_patch_map, patch_sklearn, sklearn_is_patched, unpatch_sklearn
from sklearnex.basic_statistics import BasicStatistics, IncrementalBasicStatistics
from sklearnex.linear_model import LogisticRegression
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    NearestNeighbors,
)
from sklearnex.svm import SVC, NuSVC


def _load_all_models(with_sklearnex=True, estimator=True):
    """Convert sklearnex patch_map into a dictionary of estimators or functions

    Parameters
    ----------
    with_sklearnex: bool (default=True)
        Discover estimators and methods with sklearnex patching enabled (True)
        or disabled (False) from the sklearnex patch_map

    estimator: bool (default=True)
        yield estimators (True) or functions (False)

    Returns
    -------
    dict: {name:estimator}
        estimator is a class or function from sklearn or sklearnex
    """
    # insure that patch state is correct as dictated by patch_sklearn boolean
    # and return it to the previous state no matter what occurs.
    already_patched_map = sklearn_is_patched(return_map=True)
    already_patched = any(already_patched_map.values())
    try:
        if with_sklearnex:
            patch_sklearn()
        elif already_patched:
            unpatch_sklearn()

        models = {}
        for patch_infos in get_patch_map().values():
            candidate = getattr(patch_infos[0][0][0], patch_infos[0][0][1], None)
            if candidate is not None and isclass(candidate) == estimator:
                if not estimator or issubclass(candidate, BaseEstimator):
                    models[patch_infos[0][0][1]] = candidate
    finally:
        if with_sklearnex:
            unpatch_sklearn()
        # both branches are now in an unpatched state, repatch as necessary
        if already_patched:
            patch_sklearn(name=[i for i in already_patched_map if already_patched_map[i]])

    return models


PATCHED_MODELS = _load_all_models(with_sklearnex=True)
UNPATCHED_MODELS = _load_all_models(with_sklearnex=False)

PATCHED_FUNCTIONS = _load_all_models(with_sklearnex=True, estimator=False)
UNPATCHED_FUNCTIONS = _load_all_models(with_sklearnex=False, estimator=False)

mixin_map = [
    [
        ClassifierMixin,
        ["decision_function", "predict", "predict_proba", "predict_log_proba", "score"],
        "classification",
    ],
    [RegressorMixin, ["predict", "score"], "regression"],
    [ClusterMixin, ["fit_predict"], "classification"],
    [TransformerMixin, ["fit_transform", "transform", "score"], "classification"],
    [OutlierMixin, ["fit_predict", "predict"], "classification"],
    [KNeighborsMixin, ["kneighbors"], None],
]


class sklearn_clone_dict(dict):
    """Special dict type for returning state-free sklearn/sklearnex estimators
    with the same parameters"""

    def __getitem__(self, key):
        return clone(super().__getitem__(key))


# Special dictionary of sklearnex estimators which must be specifically tested, this
# could be because of supported non-default parameters, blocked support via sklearn's
# 'available_if' decorator, or not being a native sklearn estimator (i.e. those not in
# the default PATCHED_MODELS dictionary)
SPECIAL_INSTANCES = sklearn_clone_dict(
    {
        str(i): i
        for i in [
            LocalOutlierFactor(novelty=True),
            SVC(probability=True),
            NuSVC(probability=True),
            KNeighborsClassifier(algorithm="brute"),
            KNeighborsRegressor(algorithm="brute"),
            NearestNeighbors(algorithm="brute"),
            LogisticRegression(solver="newton-cg"),
            BasicStatistics(),
            IncrementalBasicStatistics(),
        ]
    }
)


def gen_models_info(algorithms, required_inputs=["X", "y"], fit=False, daal4py=True):
    """Generate estimator-attribute pairs for pytest test collection.

    Parameters
    ----------
    algorithms : iterable (list, tuple, 1D array-like object)
        Iterable of valid sklearnex estimators or keys from PATCHED_MODELS

    required_inputs : list, tuple of strings or None
        list of required args/kwargs for callable attribute (only non-private,
        non-BaseEstimator attributes).  Only one must be present, None
        signifies taking all non-private attributes, callable or not.

    fit: bool (default False)
        Include "fit" method as an estimator-attribute pair

    daal4py: bool (default True)
        Include daal4py estimators in estimator-attribute list

    Returns
    -------
    list of 2-element tuples: (estimator, string)
        Returns a list of valid methods or attributes without "fit"
    """
    output = []
    for estimator in algorithms:

        if estimator in PATCHED_MODELS:
            est = PATCHED_MODELS[estimator]
        elif estimator in SPECIAL_INSTANCES:
            est = SPECIAL_INSTANCES[estimator].__class__
        elif isinstance(algorithms[estimator], BaseEstimator):
            est = algorithms[estimator].__class__
        else:
            raise KeyError(f"Unrecognized sklearnex estimator: {estimator}")

        if not daal4py and est.__module__.startswith("daal4py"):
            continue

        # remove BaseEstimator methods (get_params, set_params)
        candidates = set(dir(est)) - set(dir(BaseEstimator))
        # remove private methods
        candidates = set([attr for attr in candidates if not attr.startswith("_")])
        # required to enable other methods
        if not fit:
            candidates = candidates - {"fit"}

        # allow only callable methods with any of the required inputs
        if required_inputs:
            methods = []
            for attr in candidates:
                attribute = getattr_static(est, attr)
                if callable(attribute):
                    params = signature(attribute).parameters
                    if any([inp in params for inp in required_inputs]):
                        methods += [attr]
        else:
            methods = candidates

        output += (
            [(estimator, method) for method in methods]
            if methods
            else [(estimator, None)]
        )

    # In the case that no methods are available, set method to None.
    # This will allow estimators without mixins to still test the fit
    # method in various tests.
    return output


def call_method(estimator, method, X, y, **kwargs):
    """Generalized interface to call most sklearn estimator methods

    Parameters
    ----------
    estimator : sklearn or sklearnex estimator instance

    method: string
        Valid callable method to estimator

    X: array-like
        data

    y: array-like (for 'score', 'partial-fit', and 'path')
        X-dependent data

    **kwargs: keyword dict
        keyword arguments to estimator.method

    Returns
    -------
    return value from estimator.method
    """
    # useful for repository wide testing

    func = getattr(estimator, method)
    argdict = signature(func).parameters
    argnum = len(
        [i for i in argdict if argdict[i].default == Parameter.empty or i in ["X", "y"]]
    )

    if method == "inverse_transform":
        # PCA's inverse_transform takes (n_samples, n_components)
        data = (
            (X[:, : estimator.n_components_],)
            if X.shape[1] != estimator.n_components_
            else (X,)
        )
    else:
        data = (X, y)[:argnum]

    return func(*data, **kwargs)


def _gen_dataset_type(est):
    # est should be an estimator or estimator class
    # dataset initialized to classification, but will be swapped
    # for other types as necessary. Private method.
    dataset = "classification"
    estimator = est.__class__ if isinstance(est, BaseEstimator) else est

    for mixin, _, data in mixin_map:
        if issubclass(estimator, mixin) and data is not None:
            dataset = data
    return dataset


_dataset_dict = {
    "classification": [partial(load_iris, return_X_y=True)],
    "regression": [partial(load_diabetes, return_X_y=True)],
}


def gen_dataset(
    est,
    datasets=_dataset_dict,
    sparse=False,
    queue=None,
    target_df=None,
    dtype=None,
):
    """Generate dataset for pytest testing.

    Parameters
    ----------
    est : sklearn or sklearnex estimator class
        Must inherit an sklearn Mixin or sklearn's BaseEstimator

    dataset: dataset dict
        Dictionary with keys "classification" and/or "regression"
        Value must be a list of object which yield X, y array
        objects when called, ideally using a lambda or
        functools.partial.

    sparse: bool (default False)
        Convert X data to a scipy.sparse csr_matrix format.

    queue: SYCL queue or None
        Queue necessary for device offloading following the
        SYCL 2020 standard, usually generated by dpctl.

    target_df: string or None
        dataframe type for returned dataset, as dictated by
        onedal's _convert_to_dataframe.

    dtype: numpy dtype or None
       target datatype for returned datasets (see DTYPES).

    Returns
    -------
    list of 2-element list X,y: (array-like, array-like)
        list of datasets for analysis
    """
    dataset_type = _gen_dataset_type(est)
    output = []
    # load data
    flag = dtype is None

    for func in datasets[dataset_type]:
        X, y = func()
        if flag:
            dtype = X.dtype if hasattr(X, "dtype") else np.float64

        if sparse:
            X = sp.csr_matrix(X)
        else:
            X = _convert_to_dataframe(
                X, sycl_queue=queue, target_df=target_df, dtype=dtype
            )
            y = _convert_to_dataframe(
                y, sycl_queue=queue, target_df=target_df, dtype=dtype
            )
        output += [[X, y]]
    return output


def gen_sparse_dataset(row_count, column_count, **kwargs):
    """Generate sparse dataset for pytest testing.

    Parameters
    ----------
    row_count : number of rows in dataset

    column_count: number of columns in dataset

    kwargs: keyword arguments for scipy.sparse.random_array or scipy.sparse.random

    Returns
    -------
    scipy.sparse random matrix or array depending on scipy version
    """
    if hasattr(sp, "random_array"):
        return sp.random_array((row_count, column_count), **kwargs)
    else:
        return sp.random(row_count, column_count, **kwargs)


DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.float32,
    np.float64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
]


def _get_processor_info():
    proc = ""
    if platform.system() == "Linux":
        proc = (
            subprocess.check_output(["/usr/bin/cat", "/proc/cpuinfo"])
            .strip()
            .decode("utf-8")
        )
    elif platform.system() == "Windows":
        proc = platform.processor()
    elif platform.system() == "Darwin":
        proc = (
            subprocess.check_output(["/usr/bin/sysctl", "-n", "machdep.cpu.brand_string"])
            .strip()
            .decode("utf-8")
        )

    return proc


class DummyEstimator(BaseEstimator):

    def fit(self, X, y=None):
        sua_iface, xp, _ = _get_sycl_namespace(X)
        X_table = to_table(X)
        y_table = to_table(y)
        # The presence of the fitted attributes (ending with a trailing
        # underscore) is required for the correct check. The cleanup of
        # the memory will occur at the estimator instance deletion.
        if sua_iface:
            self.x_attr_ = from_table(
                X_table, sua_iface=sua_iface, sycl_queue=X.sycl_queue, xp=xp
            )
            self.y_attr_ = from_table(
                y_table,
                sua_iface=sua_iface,
                sycl_queue=X.sycl_queue if y is None else y.sycl_queue,
                xp=xp,
            )
        else:
            self.x_attr_ = from_table(X_table)
            self.y_attr_ = from_table(y_table)

        return self

    def predict(self, X):
        # Checks if the estimator is fitted by verifying the presence of
        # fitted attributes (ending with a trailing underscore).
        check_is_fitted(self)
        sua_iface, xp, _ = _get_sycl_namespace(X)
        X_table = to_table(X)
        if sua_iface:
            returned_X = from_table(
                X_table, sua_iface=sua_iface, sycl_queue=X.sycl_queue, xp=xp
            )
        else:
            returned_X = from_table(X_table)

        return returned_X

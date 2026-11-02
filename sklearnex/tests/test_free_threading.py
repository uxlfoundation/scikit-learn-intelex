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

import importlib.util
import os
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

IS_FREE_THREADED = sysconfig.get_config_var("Py_GIL_DISABLED") == 1
pytestmark = pytest.mark.skipif(
    not IS_FREE_THREADED, reason="requires a free-threaded CPython build"
)


def test_native_imports_keep_gil_disabled():
    # PYTHON_GIL must not be set here: it would keep the GIL disabled even for
    # a module that does not declare free-threading support, which is exactly
    # the regression this test is meant to catch. With '-W error', the
    # RuntimeWarning that CPython emits when it re-enables the GIL fails the
    # import.
    code = """
import importlib
import sys

assert not sys._is_gil_enabled()
importlib.import_module({module!r})
assert not sys._is_gil_enabled()
"""
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}

    dpc_backend = "onedal._onedal_py_dpc"
    native_backend = (
        dpc_backend
        if importlib.util.find_spec(dpc_backend) is not None
        else "onedal._onedal_py_host"
    )
    modules = [
        "daal4py._daal4py",
        native_backend,
        "daal4py",
        "onedal",
        "sklearnex",
    ]
    if importlib.util.find_spec("onedal._onedal_py_spmd_dpc") is not None:
        modules.append("onedal._onedal_py_spmd_dpc")

    for module in modules:
        subprocess.run(
            [sys.executable, "-W", "error", "-c", code.format(module=module)],
            check=True,
            env=env,
        )


def _fit_and_predict(name):
    from sklearnex.basic_statistics import BasicStatistics
    from sklearnex.cluster import DBSCAN, KMeans
    from sklearnex.decomposition import PCA
    from sklearnex.ensemble import RandomForestClassifier
    from sklearnex.linear_model import LinearRegression, Ridge
    from sklearnex.neighbors import KNeighborsClassifier, NearestNeighbors
    from sklearnex.svm import SVC

    X, y = make_classification(300, 8, random_state=0)
    Xr, yr = make_regression(300, 8, random_state=0)
    if name == "LinearRegression":
        return LinearRegression().fit(Xr, yr).predict(Xr)
    if name == "Ridge":
        return Ridge().fit(Xr, yr).predict(Xr)
    if name == "PCA":
        return np.abs(PCA(3).fit(X).transform(X))
    if name == "KMeans":
        return KMeans(4, n_init=1, random_state=0).fit(X).cluster_centers_
    if name == "DBSCAN":
        return DBSCAN(eps=3.0).fit(X).labels_
    if name == "RandomForestClassifier":
        return RandomForestClassifier(8, random_state=0).fit(X, y).predict_proba(X)
    if name == "KNeighborsClassifier":
        return KNeighborsClassifier().fit(X, y).predict_proba(X)
    if name == "NearestNeighbors":
        return NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)[1]
    if name == "SVC":
        return SVC().fit(X, y).decision_function(X)
    if name == "BasicStatistics":
        return BasicStatistics().fit(X).mean_
    raise KeyError(name)


_ESTIMATORS = [
    "LinearRegression",
    "Ridge",
    "PCA",
    "KMeans",
    "DBSCAN",
    "RandomForestClassifier",
    "KNeighborsClassifier",
    "NearestNeighbors",
    "SVC",
    "BasicStatistics",
]


@pytest.mark.filterwarnings("ignore:'Threading' parallel backend:UserWarning")
def test_independent_estimators_run_concurrently():
    """Separate estimator instances, each used by one thread, give the same
    results as when run serially - the supported mode in parallelism.rst."""
    expected = {name: _fit_and_predict(name) for name in _ESTIMATORS}
    tasks = _ESTIMATORS * 2
    start = Barrier(len(tasks))

    def run(name):
        start.wait()
        return name, _fit_and_predict(name)

    for _ in range(3):
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            for name, result in executor.map(run, tasks):
                np.testing.assert_allclose(result, expected[name], err_msg=name)

    assert not sys._is_gil_enabled()

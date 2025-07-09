# ===============================================================================
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
# ===============================================================================

import sklearnex
from daal4py.sklearn._utils import daal_check_version

# General use of patch_sklearn and unpatch_sklearn in pytest is not recommended.
# It changes global state and can impact the operation of other tests. This file
# specifically tests patch_sklearn and unpatch_sklearn and is exempt from this.
# If sklearnex patching is necessary in testing, use the 'with_sklearnex' pytest
# fixture.


def test_monkey_patching():
    _tokens = sklearnex.get_patch_names()
    _values = sklearnex.get_patch_map().values()
    _classes = list()

    for v in _values:
        for c in v:
            _classes.append(c[0])

    try:
        sklearnex.patch_sklearn()

        for i, _ in enumerate(_tokens):
            t = _tokens[i]
            p = _classes[i][0]
            n = _classes[i][1]

            class_module = getattr(p, n).__module__
            assert any(
                class_module.startswith(prefix) for prefix in ["daal4py", "sklearnex"]
            ), "Patching has completed with error."

        for i, _ in enumerate(_tokens):
            t = _tokens[i]
            p = _classes[i][0]
            n = _classes[i][1]

            sklearnex.unpatch_sklearn(t)
            sklearn_class = getattr(p, n, None)
            if sklearn_class is not None:
                sklearn_class = sklearn_class.__module__
            assert sklearn_class is None or sklearn_class.startswith(
                "sklearn"
            ), "Unpatching has completed with error."

    finally:
        sklearnex.unpatch_sklearn()

    try:
        for i, _ in enumerate(_tokens):
            t = _tokens[i]
            p = _classes[i][0]
            n = _classes[i][1]

            sklearn_class = getattr(p, n, None)
            if sklearn_class is not None:
                sklearn_class = sklearn_class.__module__
            assert sklearn_class is None or sklearn_class.startswith(
                "sklearn"
            ), "Unpatching has completed with error."

    finally:
        sklearnex.unpatch_sklearn()

    try:
        for i, _ in enumerate(_tokens):
            t = _tokens[i]
            p = _classes[i][0]
            n = _classes[i][1]

            sklearnex.patch_sklearn(t)

            class_module = getattr(p, n).__module__
            assert any(
                class_module.startswith(prefix) for prefix in ["daal4py", "sklearnex"]
            ), "Patching has completed with error."
    finally:
        sklearnex.unpatch_sklearn()


def test_patch_by_list_simple():
    try:
        sklearnex.patch_sklearn(["LogisticRegression"])

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVC

        assert RandomForestRegressor.__module__.startswith("sklearn")
        assert KNeighborsRegressor.__module__.startswith("sklearn")
        if daal_check_version((2024, "P", 1)):
            assert LogisticRegression.__module__.startswith("sklearnex")
        else:
            assert LogisticRegression.__module__.startswith("daal4py")
        assert SVC.__module__.startswith("sklearn")
    finally:
        sklearnex.unpatch_sklearn()


def test_patch_by_list_many_estimators():
    try:
        sklearnex.patch_sklearn(["LogisticRegression", "SVC"])

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVC

        assert RandomForestRegressor.__module__.startswith("sklearn")
        assert KNeighborsRegressor.__module__.startswith("sklearn")
        if daal_check_version((2024, "P", 1)):
            assert LogisticRegression.__module__.startswith("sklearnex")
        else:
            assert LogisticRegression.__module__.startswith("daal4py")
        assert SVC.__module__.startswith("daal4py") or SVC.__module__.startswith(
            "sklearnex"
        )

    finally:
        sklearnex.unpatch_sklearn()


def test_unpatch_by_list_many_estimators():
    try:
        sklearnex.patch_sklearn()

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVC

        assert RandomForestRegressor.__module__.startswith("sklearnex")
        assert KNeighborsRegressor.__module__.startswith(
            "daal4py"
        ) or KNeighborsRegressor.__module__.startswith("sklearnex")
        if daal_check_version((2024, "P", 1)):
            assert LogisticRegression.__module__.startswith("sklearnex")
        else:
            assert LogisticRegression.__module__.startswith("daal4py")
        assert SVC.__module__.startswith("daal4py") or SVC.__module__.startswith(
            "sklearnex"
        )

        sklearnex.unpatch_sklearn(["KNeighborsRegressor", "RandomForestRegressor"])

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsRegressor
        from sklearn.svm import SVC

        assert RandomForestRegressor.__module__.startswith("sklearn")
        assert KNeighborsRegressor.__module__.startswith("sklearn")
        if daal_check_version((2024, "P", 1)):
            assert LogisticRegression.__module__.startswith("sklearnex")
        else:
            assert LogisticRegression.__module__.startswith("daal4py")

        assert SVC.__module__.startswith("daal4py") or SVC.__module__.startswith(
            "sklearnex"
        )
    finally:
        sklearnex.unpatch_sklearn()


def test_patching_checker():
    for name in [None, "SVC", "PCA"]:
        try:
            sklearnex.patch_sklearn(name=name)
            assert sklearnex.sklearn_is_patched(name=name)

        finally:
            sklearnex.unpatch_sklearn(name=name)
            assert not sklearnex.sklearn_is_patched(name=name)
    try:
        sklearnex.patch_sklearn()
        patching_status_map = sklearnex.sklearn_is_patched(return_map=True)
        assert len(patching_status_map) == len(sklearnex.get_patch_names())
        for status in patching_status_map.values():
            assert status
    finally:
        sklearnex.unpatch_sklearn()

    patching_status_map = sklearnex.sklearn_is_patched(return_map=True)
    assert len(patching_status_map) == len(sklearnex.get_patch_names())
    for status in patching_status_map.values():
        assert not status


def test_preview_namespace():
    def get_estimators():
        from sklearn.cluster import DBSCAN
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LinearRegression
        from sklearn.svm import SVC

        return (
            LinearRegression(),
            PCA(),
            DBSCAN(),
            SVC(),
            RandomForestClassifier(),
        )

    from sklearnex.dispatcher import _is_preview_enabled

    try:
        sklearnex.patch_sklearn(preview=True)

        assert _is_preview_enabled()

        lr, pca, dbscan, svc, rfc = get_estimators()
        assert "sklearnex" in rfc.__module__

        if daal_check_version((2023, "P", 100)):
            assert "sklearnex" in lr.__module__
        else:
            assert "daal4py" in lr.__module__

        assert "sklearnex" in pca.__module__
        assert "sklearnex" in dbscan.__module__
        assert "sklearnex" in svc.__module__

    finally:
        sklearnex.unpatch_sklearn()

    # no patching behavior
    lr, pca, dbscan, svc, rfc = get_estimators()
    assert "sklearn." in lr.__module__ and "daal4py" not in lr.__module__
    assert "sklearn." in pca.__module__ and "daal4py" not in pca.__module__
    assert "sklearn." in dbscan.__module__ and "daal4py" not in dbscan.__module__
    assert "sklearn." in svc.__module__ and "daal4py" not in svc.__module__
    assert "sklearn." in rfc.__module__ and "daal4py" not in rfc.__module__

    # default patching behavior
    try:
        sklearnex.patch_sklearn()
        assert not _is_preview_enabled()

        lr, pca, dbscan, svc, rfc = get_estimators()

        if daal_check_version((2023, "P", 100)):
            assert "sklearnex" in lr.__module__
        else:
            assert "daal4py" in lr.__module__

        assert "sklearnex" in pca.__module__
        assert "sklearnex" in rfc.__module__
        assert "sklearnex" in dbscan.__module__
        assert "sklearnex" in svc.__module__
    finally:
        sklearnex.unpatch_sklearn()

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

import pytest
from numpy.testing import assert_allclose

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    LocalOutlierFactor,
    NearestNeighbors,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_knn_classifier(dataframe, queue):
    import sys
    print(f"\n=== DEBUG test_sklearnex_import_knn_classifier START: dataframe={dataframe}, queue={queue} ===", file=sys.stderr)
    X = _convert_to_dataframe([[0], [1], [2], [3]], sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
    y = _convert_to_dataframe([0, 0, 1, 1], sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: y type={type(y)}", file=sys.stderr)
    print(f"DEBUG test: Creating KNeighborsClassifier and calling fit", file=sys.stderr)
    neigh = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    print(f"DEBUG test: fit completed, neigh._fit_X type={type(getattr(neigh, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
    y_test = _convert_to_dataframe([[1.1]], sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: Calling predict with y_test type={type(y_test)}", file=sys.stderr)
    pred = _as_numpy(neigh.predict(y_test))
    print(f"DEBUG test: predict completed, pred={pred}", file=sys.stderr)
    assert "sklearnex" in neigh.__module__
    assert_allclose(pred, [0])
    print(f"=== DEBUG test_sklearnex_import_knn_classifier END ===\n", file=sys.stderr)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_knn_regression(dataframe, queue):
    import sys
    print(f"\n=== DEBUG test_sklearnex_import_knn_regression START: dataframe={dataframe}, queue={queue} ===", file=sys.stderr)
    X = _convert_to_dataframe([[0], [1], [2], [3]], sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
    y = _convert_to_dataframe([0, 0, 1, 1], sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: y type={type(y)}", file=sys.stderr)
    print(f"DEBUG test: Creating KNeighborsRegressor and calling fit", file=sys.stderr)
    neigh = KNeighborsRegressor(n_neighbors=2).fit(X, y)
    print(f"DEBUG test: fit completed, neigh._fit_X type={type(getattr(neigh, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
    y_test = _convert_to_dataframe([[1.5]], sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: Calling predict with y_test type={type(y_test)}", file=sys.stderr)
    pred = _as_numpy(neigh.predict(y_test)).squeeze()
    print(f"DEBUG test: predict completed, pred={pred}", file=sys.stderr)
    assert "sklearnex" in neigh.__module__
    assert_allclose(pred, 0.5)
    print(f"=== DEBUG test_sklearnex_import_knn_regression END ===\n", file=sys.stderr)


@pytest.mark.parametrize("algorithm", ["auto", "brute"])
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize(
    "estimator",
    [LocalOutlierFactor, NearestNeighbors],
)
def test_sklearnex_kneighbors(algorithm, estimator, dataframe, queue):
    import sys
    print(f"\n=== DEBUG test_sklearnex_kneighbors START: algorithm={algorithm}, estimator={estimator.__name__}, dataframe={dataframe}, queue={queue} ===", file=sys.stderr)
    X = [[0, 0, 2], [1, 0, 0], [0, 0, 1]]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
    test = _convert_to_dataframe([[0, 0, 1.3]], sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: test type={type(test)}", file=sys.stderr)
    print(f"DEBUG test: Creating {estimator.__name__} and calling fit", file=sys.stderr)
    neigh = estimator(n_neighbors=2, algorithm=algorithm).fit(X)
    print(f"DEBUG test: fit completed, neigh._fit_X type={type(getattr(neigh, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
    print(f"DEBUG test: Calling kneighbors", file=sys.stderr)
    result = neigh.kneighbors(test, 2, return_distance=False)
    result = _as_numpy(result)
    print(f"DEBUG test: kneighbors completed, result={result}", file=sys.stderr)
    assert "sklearnex" in neigh.__module__
    assert_allclose(result, [[2, 0]])
    print(f"DEBUG test: Calling kneighbors with no args", file=sys.stderr)
    result = neigh.kneighbors()
    print(f"=== DEBUG test_sklearnex_kneighbors END ===\n", file=sys.stderr)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_lof(dataframe, queue):
    import sys
    print(f"\n=== DEBUG test_sklearnex_import_lof START: dataframe={dataframe}, queue={queue} ===", file=sys.stderr)
    X = [[7, 7, 7], [1, 0, 0], [0, 0, 1], [0, 0, 1]]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    print(f"DEBUG test: X type={type(X)}, X shape={getattr(X, 'shape', 'NO_SHAPE')}", file=sys.stderr)
    print(f"DEBUG test: Creating LocalOutlierFactor and calling fit_predict", file=sys.stderr)
    lof = LocalOutlierFactor(n_neighbors=2)
    result = lof.fit_predict(X)
    result = _as_numpy(result)
    print(f"DEBUG test: fit_predict completed, result={result}", file=sys.stderr)
    print(f"DEBUG test: lof._fit_X type={type(getattr(lof, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
    assert hasattr(lof, "_onedal_estimator")
    assert "sklearnex" in lof.__module__
    assert_allclose(result, [-1, 1, 1, 1])
    print(f"=== DEBUG test_sklearnex_import_lof END ===\n", file=sys.stderr)
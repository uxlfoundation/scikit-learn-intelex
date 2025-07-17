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

import numpy as np
import pytest
from numpy.testing import assert_allclose

try:
    from scipy.sparse import csr_array as csr_class
except ImportError:
    from scipy.sparse import csr_matrix as csr_class

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_svc(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("SVC fit for the GPU sycl_queue is buggy.")
    from sklearnex.svm import SVC

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = SVC(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(_as_numpy(svc.dual_coef_), [[-0.25, 0.25]])
    assert_allclose(_as_numpy(svc.support_), [1, 3])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_nusvc(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("NuSVC fit for the GPU sycl_queue is buggy.")
    from sklearnex.svm import NuSVC

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = NuSVC(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(
        _as_numpy(svc.dual_coef_), [[-0.04761905, -0.0952381, 0.0952381, 0.04761905]]
    )
    assert_allclose(_as_numpy(svc.support_), [0, 1, 3, 4])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_svr(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("SVR fit for the GPU sycl_queue is buggy.")
    from sklearnex.svm import SVR

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = SVR(kernel="linear").fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(_as_numpy(svc.dual_coef_), [[-0.1, 0.1]])
    assert_allclose(_as_numpy(svc.support_), [1, 3])


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_nusvr(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("NuSVR fit for the GPU sycl_queue is buggy.")
    from sklearnex.svm import NuSVR

    X = np.array([[-2, -1], [-1, -1], [-1, -2], [+1, +1], [+1, +2], [+2, +1]])
    y = np.array([1, 1, 1, 2, 2, 2])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    svc = NuSVR(kernel="linear", nu=0.9).fit(X, y)
    assert "daal4py" in svc.__module__ or "sklearnex" in svc.__module__
    assert_allclose(
        _as_numpy(svc.dual_coef_), [[-1.0, 0.611111, 1.0, -0.611111]], rtol=1e-3
    )
    assert_allclose(_as_numpy(svc.support_), [1, 2, 3, 5])


# https://github.com/uxlfoundation/scikit-learn-intelex/issues/1880
def test_works_with_unsorted_indices():
    from sklearnex.svm import SVC

    X = csr_class(
        (
            np.array(
                [0.44943642, 0.6316672, 0.6316672, 0.44943642, 0.6316672, 0.6316672]
            ),
            np.array([1, 4, 3, 1, 2, 0], dtype=np.int32),
            np.array([0, 3, 6], dtype=np.int32),
        ),
        shape=(2, 5),
    )
    y = np.array([1, 0])
    X_test_single = np.array([[1, 0, 0, 0, 0]], dtype=np.float64)
    X_test_multi = np.array([[1, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.float64)
    model = SVC(probability=True).fit(X, y)
    pred_single = model.predict_proba(X_test_single)
    pred_multi = model.predict_proba(X_test_multi)[0]
    np.testing.assert_array_equal(
        pred_single.reshape(-1),
        pred_multi.reshape(-1),
    )

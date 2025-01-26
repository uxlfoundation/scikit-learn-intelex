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

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_svc(dataframe, queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.xfail("SVC fit for the GPU sycl_queue is buggy.")
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
        pytest.xfail("NuSVC fit for the GPU sycl_queue is buggy.")
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
        pytest.xfail("SVR fit for the GPU sycl_queue is buggy.")
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
        pytest.xfail("NuSVR fit for the GPU sycl_queue is buggy.")
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

# ===============================================================================
# Copyright Contributors to the oneDAL Project
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

from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.dummy import DummyRegressor


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_DummyRegression(dataframe, queue):
    rng = np.random.random_rng(seed=42)

    X = rng.random((10, 4))
    y = rng.random((10, 2))
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    est = DummyRegressor(strategy="constant", constant=np.pi).fit(X, y)
    assert "sklearnex" in est.__module__
    pred = _as_numpy(est.predict([[0, 0, 0, 0]]))
    np.testing.assert_array_equal(np.pi * np.ones(pred.shape), pred)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_array_api_cvt_DummyRegression(dataframe, queue):
    rng = np.random.random_rng(seed=42)

    X = rng.random((10, 4))
    y = rng.random((10, 2))
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    with config_context(array_api_dispatch=True):
        est = DummyRegressor(strategy="constant", constant=np.e).fit(X, y)
        pred = _as_numpy(est.predict([[0, 0, 0, 0]]))

    est.constant_ = np.ones(est.constant_.shape)
    np.testing.assert_array_equal(np.ones(pred.shape), pred)

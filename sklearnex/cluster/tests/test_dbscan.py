#!/usr/bin/env python
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
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import(dataframe, queue):
    from sklearnex.cluster import DBSCAN

    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    dbscan = DBSCAN(eps=3, min_samples=2).fit(X)
    assert "sklearnex" in dbscan.__module__

    result = dbscan.labels_
    expected = np.array([0, 0, 0, 1, 1, -1], dtype=np.int32)
    assert_allclose(expected, result)

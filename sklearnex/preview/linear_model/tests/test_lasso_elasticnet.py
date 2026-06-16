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

import os

import pytest
from numpy.testing import assert_allclose
from sklearn.datasets import make_regression

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


# Note: Lasso and ElasticNet do not have GPU implementations.
# If that changes, then the filters for 'get_dataframes_and_queues'
# should be removed from these tests.
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("numpy,pandas", "cpu")
)
def test_sklearnex_import_lasso(dataframe, queue):
    from sklearnex.preview.linear_model import Lasso

    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    lasso = Lasso(alpha=0.1).fit(X, y)
    assert "daal4py" in lasso.__module__
    assert_allclose(lasso.intercept_, 0.15)
    assert_allclose(lasso.coef_, [0.85, 0.0])


@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("numpy,pandas", "cpu")
)
def test_sklearnex_import_elastic(dataframe, queue):
    from sklearnex.preview.linear_model import ElasticNet

    X, y = make_regression(n_features=2, random_state=0)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    elasticnet = ElasticNet(random_state=0).fit(X, y)
    assert "daal4py" in elasticnet.__module__
    assert_allclose(elasticnet.intercept_, 1.451, atol=1e-3)
    assert_allclose(elasticnet.coef_, [18.838, 64.559], atol=1e-3)

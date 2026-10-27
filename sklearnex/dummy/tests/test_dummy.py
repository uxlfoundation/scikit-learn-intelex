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
    skip_array_api_strict_readonly,
)
from sklearnex import config_context
from sklearnex.dummy import DummyRegressor


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_DummyRegressor(dataframe, queue):
    skip_array_api_strict_readonly(dataframe)
    rng = np.random.default_rng(seed=42)

    X = rng.random((10, 4))
    y = rng.random((10,))
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    est = DummyRegressor(strategy="constant", constant=np.pi).fit(X, y)
    assert "sklearnex" in est.__module__
    X_test = _convert_to_dataframe([[0, 0, 0, 0]], sycl_queue=queue, target_df=dataframe)
    pred = _as_numpy(est.predict(X_test))
    np.testing.assert_array_equal(np.pi * np.ones(pred.shape), pred)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("dpnp"))
def test_fitted_attribute_conversion_DummyRegressor(dataframe, queue, with_array_api):
    rng = np.random.default_rng(seed=42)

    X = rng.random((10, 4))
    y = rng.random((10,))
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    X_test = _convert_to_dataframe([[0, 0, 0, 0]], sycl_queue=queue, target_df=dataframe)
    est = DummyRegressor(strategy="constant", constant=np.e).fit(X, y)
    pred = _as_numpy(est.predict(X_test))

    np.testing.assert_array_equal(np.full(pred.shape, np.e), pred)
    est.constant_ = np.ones(est.constant_.shape)
    np.testing.assert_array_equal(np.ones(pred.shape), _as_numpy(est.predict(X_test)))


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize("n_targets", [1, 2])
def test_model_recreation_after_fallback(n_targets):
    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(20, 3))
    y = rng.standard_gamma(1, size=(X.shape[0], n_targets))
    if y.shape[1] == 1:
        y = y.reshape(-1)

    # Strategy 'median' is not supported by oneDAL, so this
    # should fall back to scikit-learn
    estimator = DummyRegressor(strategy="median").fit(X, y)
    assert not hasattr(estimator, "_onedal_estimator")

    # During predictions, it should be able to create a oneDAL
    # estimator on-the-fly based on scikit-learn's attributes
    # when predictions are supported through oneDAL
    pred = estimator.predict(X)
    if n_targets == 1:
        assert hasattr(estimator, "_onedal_estimator")

    reference = np.tile(np.median(y, axis=0), X.shape[0])
    if n_targets > 1:
        reference = reference.reshape((X.shape[0], -1))
    np.testing.assert_almost_equal(pred, reference)

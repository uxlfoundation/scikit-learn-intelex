# ==============================================================================
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
# ==============================================================================

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import sparse as sp
from sklearn.preprocessing import MaxAbsScaler as _sklearn_MaxAbsScaler

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.preview.preprocessing import MaxAbsScaler


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_MaxAbsScaler(dataframe, queue):
    # Verify that the estimator gets properly imported from sklearnex
    rng = np.random.default_rng(seed=42)
    X = rng.random((10, 4))
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    est = MaxAbsScaler().fit(X)
    assert "sklearnex" in est.__module__


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_max_abs_scaler_dense_fit_transform(dataframe, queue):
    # Test parity with scikit-learn for basic fit_transform behavior
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((50, 5))

    # Randomly scale some columns to have varying absolute max values
    X[:, 0] *= 10
    X[:, 1] *= 0.1
    X[:, 2] += 5

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    # Scikit-learn Baseline
    scaler_sk = _sklearn_MaxAbsScaler()
    X_trans_sk = scaler_sk.fit_transform(X)

    # Sklearnex
    scaler_ex = MaxAbsScaler()
    X_trans_ex = scaler_ex.fit_transform(X_df)
    X_trans_ex_np = _as_numpy(X_trans_ex)

    assert_allclose(scaler_ex.scale_, scaler_sk.scale_)
    assert_allclose(scaler_ex.max_abs_, scaler_sk.max_abs_)
    assert_allclose(X_trans_ex_np, X_trans_sk)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_max_abs_scaler_dense_partial_fit(dataframe, queue):
    # Test batch processing parity with native scikit-learn
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((100, 3))

    # create batches
    X1, X2, X3 = X[:30], X[30:70], X[70:]

    # Scikit-learn baseline
    scaler_sk = _sklearn_MaxAbsScaler()
    for batch in [X1, X2, X3]:
        scaler_sk.partial_fit(batch)
    X_trans_sk = scaler_sk.transform(X)

    # Sklearnex execution
    scaler_ex = MaxAbsScaler()
    for batch in [X1, X2, X3]:
        batch_df = _convert_to_dataframe(batch, sycl_queue=queue, target_df=dataframe)
        scaler_ex.partial_fit(batch_df)

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    X_trans_ex = scaler_ex.transform(X_df)
    X_trans_ex_np = _as_numpy(X_trans_ex)

    assert scaler_ex.n_samples_seen_ == scaler_sk.n_samples_seen_
    assert_allclose(scaler_ex.scale_, scaler_sk.scale_)
    assert_allclose(scaler_ex.max_abs_, scaler_sk.max_abs_)
    assert_allclose(X_trans_ex_np, X_trans_sk)


@pytest.mark.skipif(
    not sklearn_check_version("1.3"), reason="lacks sklearn array API support"
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues("dpctl,dpnp"))
def test_max_abs_scaler_array_api_dispatch(dataframe, queue):
    # Ensure properties are properly constructed as the dispatched arrays using Array API
    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((10, 4))

    X_df = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        est = MaxAbsScaler().fit(X_df)
        X_trans = est.transform(X_df)

    # Verify the property types respect array api execution outputs.
    # The scale_ out typically relies on standard numpy if DPCTL/DPNP isn't requested natively
    # via the context namespace, but let's just make sure it behaves normally.
    assert hasattr(est, "scale_")
    assert hasattr(est, "max_abs_")

    est.scale_ = np.ones(est.scale_.shape)
    X_trans_modified = est.transform(X_df)

    X_np = _as_numpy(X_df)
    X_trans_modified_np = _as_numpy(X_trans_modified)

    # Testing that after artificially modifying the scaler properties, the transform
    # executes normally (just returns the raw variables over 1.0 logic).
    assert_allclose(X_np, X_trans_modified_np)

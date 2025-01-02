# ==============================================================================
# Copyright 2024 Intel Corporation
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

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2024, "P", 600)):
    import numpy as np
    import pytest
    from numpy.testing import assert_allclose, assert_array_equal
    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    from onedal.datatypes import from_table
    from onedal.linear_model import IncrementalRidge
    from onedal.tests.utils._device_selection import get_queues

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_diabetes(queue, dtype):
        X, y = load_diabetes(return_X_y=True)
        X, y = X.astype(dtype), y.astype(dtype)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=777
        )
        X_train_split = np.array_split(X_train, 2)
        y_train_split = np.array_split(y_train, 2)
        model = IncrementalRidge(fit_intercept=True, alpha=0.1)
        for i in range(2):
            model.partial_fit(X_train_split[i], y_train_split[i], queue=queue)
        model.finalize_fit()
        y_pred = model.predict(X_test, queue=queue)
        assert_allclose(mean_squared_error(y_test, y_pred), 2388.775, rtol=1e-5)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.skip(reason="pickling not implemented for oneDAL entities")
    def test_pickle(queue, dtype):
        # TODO Implement pickling for oneDAL entities
        X, y = load_diabetes(return_X_y=True)
        X, y = X.astype(dtype), y.astype(dtype)
        model = IncrementalRidge(fit_intercept=True, alpha=0.5)
        model.partial_fit(X, y, queue=queue)
        model.finalize_fit()
        expected = model.predict(X, queue=queue)

        import pickle

        dump = pickle.dumps(model)
        model2 = pickle.loads(dump)

        assert isinstance(model2, model.__class__)
        result = model2.predict(X, queue=queue)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("num_blocks", [1, 2, 10])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_no_intercept_results(queue, num_blocks, dtype):
        seed = 42
        n_features, n_targets = 19, 7
        n_train_samples, n_test_samples = 3500, 1999

        gen = np.random.default_rng(seed)

        X = gen.random(size=(n_train_samples, n_features), dtype=dtype)
        y = gen.random(size=(n_train_samples, n_targets), dtype=dtype)
        X_split = np.array_split(X, num_blocks)
        y_split = np.array_split(y, num_blocks)
        alpha = 0.5

        lambda_identity = alpha * np.eye(X.shape[1])
        inverse_term = np.linalg.inv(np.dot(X.T, X) + lambda_identity)
        xt_y = np.dot(X.T, y)
        coef = np.dot(inverse_term, xt_y)

        model = IncrementalRidge(fit_intercept=False, alpha=alpha)
        for i in range(num_blocks):
            model.partial_fit(X_split[i], y_split[i], queue=queue)
        model.finalize_fit()

        if queue and queue.sycl_device.is_gpu:
            tol = 5e-3 if model.coef_.dtype == np.float32 else 1e-5
        else:
            tol = 2e-3 if model.coef_.dtype == np.float32 else 1e-5
        assert_allclose(coef, model.coef_.T, rtol=tol)

        Xt = gen.random(size=(n_test_samples, n_features), dtype=dtype)
        gtr = Xt @ coef

        res = model.predict(Xt, queue=queue)

        tol = 2e-4 if res.dtype == np.float32 else 1e-7
        assert_allclose(gtr, res, rtol=tol)

    @pytest.mark.parametrize("queue", get_queues())
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_incremental_estimator_pickle(queue, dtype):
        import pickle

        model = IncrementalRidge()

        # Check that estimator can be serialized without any data.
        dump = pickle.dumps(model)
        model_loaded = pickle.loads(dump)
        seed = 77
        gen = np.random.default_rng(seed)
        X = gen.uniform(low=-0.3, high=+0.7, size=(10, 10))
        X = X.astype(dtype)
        coef = gen.random(size=(1, 10), dtype=dtype).T
        y = X @ coef
        X_split = np.array_split(X, 2)
        y_split = np.array_split(y, 2)
        model.partial_fit(X_split[0], y_split[0], queue=queue)
        model_loaded.partial_fit(X_split[0], y_split[0], queue=queue)

        # model.finalize_fit()

        assert model._need_to_finalize == True
        assert model_loaded._need_to_finalize == True

        # Check that estimator can be serialized after partial_fit call.
        dump = pickle.dumps(model)
        model_loaded = pickle.loads(dump)

        partial_xtx = from_table(model._partial_result.partial_xtx)
        partial_xtx_loaded = from_table(model_loaded._partial_result.partial_xtx)
        assert_allclose(partial_xtx, partial_xtx_loaded)

        partial_xty = from_table(model._partial_result.partial_xty)
        partial_xty_loaded = from_table(model_loaded._partial_result.partial_xty)
        assert_allclose(partial_xty, partial_xty_loaded)

        assert model._need_to_finalize == False
        # Finalize is called during serialization to make sure partial results are finalized correctly.
        assert model_loaded._need_to_finalize == False

        model.partial_fit(X_split[1], y_split[1], queue=queue)
        model_loaded.partial_fit(X_split[1], y_split[1], queue=queue)
        assert model._need_to_finalize == True
        assert model_loaded._need_to_finalize == True

        dump = pickle.dumps(model_loaded)
        model_loaded = pickle.loads(dump)

        assert model._need_to_finalize == True
        assert model_loaded._need_to_finalize == False

        model.finalize_fit()
        model_loaded.finalize_fit()

        # Check that finalized estimator can be serialized.
        dump = pickle.dumps(model_loaded)
        model_loaded = pickle.loads(dump)

        assert_allclose(model.coef_, model_loaded.coef_, atol=1e-6)
        assert_allclose(model.intercept_, model_loaded.intercept_, atol=1e-6)

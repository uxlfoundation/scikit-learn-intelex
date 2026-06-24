# ===============================================================================
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
# ===============================================================================

from daal4py.sklearn._utils import daal_check_version

if daal_check_version((2024, "P", 600)):
    import numpy as np
    import pytest
    from numpy.testing import assert_allclose, assert_array_equal
    from sklearn.datasets import load_diabetes
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    from onedal.linear_model import Ridge
    from onedal.tests.utils._device_selection import get_queues

    @pytest.mark.parametrize(
        "queue,dtype", get_queues(dtypes=[np.float32, np.float64])
    )
    def test_diabetes(queue, dtype):
        X, y = load_diabetes(return_X_y=True)
        X, y = X.astype(dtype), y.astype(dtype)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.8, random_state=777
        )
        model = Ridge(fit_intercept=True, alpha=0.1)
        model.fit(X_train, y_train, queue=queue)
        y_pred = model.predict(X_test, queue=queue)
        assert_allclose(mean_squared_error(y_test, y_pred), 2388.775, rtol=1e-5)

    @pytest.mark.parametrize(
        "queue,dtype", get_queues(dtypes=[np.float32, np.float64])
    )
    def test_pickle(queue, dtype):
        X, y = load_diabetes(return_X_y=True)
        X, y = X.astype(dtype), y.astype(dtype)
        model = Ridge(fit_intercept=True, alpha=0.5)
        model.fit(X, y, queue=queue)
        expected = model.predict(X, queue=queue)

        import pickle

        dump = pickle.dumps(model)
        model2 = pickle.loads(dump)

        assert isinstance(model2, model.__class__)
        result = model2.predict(X, queue=queue)

        assert_array_equal(expected, result)

    @pytest.mark.parametrize(
        "queue,dtype", get_queues(dtypes=[np.float32, np.float64])
    )
    def test_no_intercept_results(queue, dtype):
        seed = 42
        n_features, n_targets = 19, 7
        n_train_samples, n_test_samples = 3500, 1999

        gen = np.random.default_rng(seed)

        X = gen.random(size=(n_train_samples, n_features), dtype=dtype)
        y = gen.random(size=(n_train_samples, n_targets), dtype=dtype)
        alpha = 0.5

        lambda_identity = alpha * np.eye(X.shape[1])
        inverse_term = np.linalg.inv(np.dot(X.T, X) + lambda_identity)
        xt_y = np.dot(X.T, y)
        coef = np.dot(inverse_term, xt_y)

        model = Ridge(fit_intercept=False, alpha=alpha)
        model.fit(X, y, queue=queue)

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

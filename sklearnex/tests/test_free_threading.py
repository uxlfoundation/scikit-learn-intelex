# ==============================================================================
# Copyright 2026 Intel Corporation
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

import importlib.util
import os
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor

import pytest

IS_FREE_THREADED = sysconfig.get_config_var("Py_GIL_DISABLED") == 1
pytestmark = pytest.mark.skipif(
    not IS_FREE_THREADED, reason="requires a free-threaded CPython build"
)


def test_native_imports_keep_gil_disabled():
    code = """
import importlib
import sys
import sysconfig

assert sysconfig.get_config_var("Py_GIL_DISABLED") == 1
assert not sys._is_gil_enabled()
importlib.import_module({module!r})
assert not sys._is_gil_enabled()
"""
    env = os.environ.copy()
    env["PYTHON_GIL"] = "0"

    dpc_backend = "onedal._onedal_py_dpc"
    native_backend = (
        dpc_backend
        if importlib.util.find_spec(dpc_backend) is not None
        else "onedal._onedal_py_host"
    )
    modules = [
        "daal4py._daal4py",
        native_backend,
        "daal4py",
        "onedal",
        "sklearnex",
    ]
    if importlib.util.find_spec("onedal._onedal_py_spmd_dpc") is not None:
        modules.append("onedal._onedal_py_spmd_dpc")

    for module in modules:
        subprocess.run(
            [sys.executable, "-W", "error", "-c", code.format(module=module)],
            check=True,
            env=env,
        )


@pytest.mark.filterwarnings(
    "ignore:'Threading' parallel backend is not supported.*:UserWarning"
)
def test_independent_linear_regressions_run_concurrently_without_gil():
    import numpy as np

    from sklearnex.linear_model import LinearRegression

    assert not sys._is_gil_enabled()
    x = np.arange(400, dtype=np.float64).reshape(200, 2)
    y = 3.0 * x[:, 0] - 2.0 * x[:, 1] + 5.0

    def fit_and_predict(_):
        model = LinearRegression().fit(x.copy(), y.copy())
        assert hasattr(model, "_onedal_estimator")
        prediction = model.predict(x[:10].copy())
        np.testing.assert_allclose(prediction, y[:10], rtol=1e-7, atol=1e-7)
        return prediction

    with ThreadPoolExecutor(max_workers=8) as executor:
        predictions = list(executor.map(fit_and_predict, range(32)))

    for prediction in predictions[1:]:
        np.testing.assert_allclose(prediction, predictions[0], rtol=0, atol=0)
    assert not sys._is_gil_enabled()


def test_shared_daal4py_algorithm_is_serialized():
    import numpy as np

    import daal4py

    x = np.arange(400, dtype=np.float64).reshape(200, 2)
    y = (3.0 * x[:, 0] - 2.0 * x[:, 1] + 5.0).reshape(-1, 1)
    training = daal4py.linear_regression_training()

    def train_and_predict(_):
        model = training.compute(x.copy(), y.copy()).model
        prediction = (
            daal4py.linear_regression_prediction()
            .compute(x[:10].copy(), model)
            .prediction
        )
        np.testing.assert_allclose(prediction[:, 0], y[:10, 0], rtol=1e-7, atol=1e-7)
        return prediction

    with ThreadPoolExecutor(max_workers=8) as executor:
        predictions = list(executor.map(train_and_predict, range(32)))

    for prediction in predictions[1:]:
        np.testing.assert_allclose(prediction, predictions[0], rtol=0, atol=0)
    assert not sys._is_gil_enabled()


def test_daal4py_model_state_is_serialized_with_readers():
    import numpy as np

    import daal4py

    x = np.arange(400, dtype=np.float64).reshape(200, 2)
    y = (x[:, 0] > x[:, 1]).astype(np.int64).reshape(-1, 1)
    model = (
        daal4py.decision_forest_classification_training(
            nClasses=2,
            nTrees=4,
        )
        .compute(x, y)
        .model
    )
    state = model.__getstate__()

    def replace_state():
        for _ in range(32):
            model.__setstate__(state)

    def read_state():
        for _ in range(32):
            assert model.NumberOfTrees == 4
            assert model.__getstate__()
            assert repr(model)
            assert daal4py.getTreeState(model, 0, 2) is not None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(replace_state)]
        futures.extend(executor.submit(read_state) for _ in range(3))
        for future in futures:
            future.result()

    assert not sys._is_gil_enabled()


@pytest.mark.parametrize("legacy_capsule", [False, True])
def test_daal4py_numeric_table_protocol_is_thread_local(legacy_capsule):
    import numpy as np

    import daal4py._daal4py as backend

    arrays = (
        np.full((32, 4), 11.0, dtype=np.float64),
        np.full((32, 4), 29.0, dtype=np.float64),
    )

    class NumericTableProtocol:
        def __init__(self, array):
            self.array = array

        def __2daalnt__(self):
            return backend._make_nt_capsule_for_testing(self.array, legacy=legacy_capsule)

    objects = tuple(NumericTableProtocol(array) for array in arrays)

    def roundtrip(index):
        expected = arrays[index % 2]
        result = backend._roundtrip_nt_for_testing(objects[index % 2])
        np.testing.assert_array_equal(result, expected)
        return result[0, 0]

    with ThreadPoolExecutor(max_workers=8) as executor:
        values = list(executor.map(roundtrip, range(1000)))

    assert values == [11.0 if index % 2 == 0 else 29.0 for index in range(1000)]
    assert not sys._is_gil_enabled()


def test_onedal_table_keeps_numpy_and_csr_owners_alive():
    import gc

    import numpy as np
    from scipy import sparse

    from onedal.datatypes._data_conversion import from_table, to_table

    dense = np.arange(256, dtype=np.float64).reshape(64, 4)
    expected_dense = dense.copy()
    dense_table = to_table(dense)
    del dense
    gc.collect()
    _ = [np.empty((64, 4), dtype=np.float64) for _ in range(128)]
    np.testing.assert_array_equal(from_table(dense_table), expected_dense)

    csr = sparse.csr_matrix(expected_dense)
    expected_csr = csr.toarray()
    csr_table = to_table(csr)
    del csr
    gc.collect()
    _ = [np.empty(expected_dense.size, dtype=np.float64) for _ in range(128)]
    csr_data, csr_indices, csr_indptr = from_table(csr_table)
    actual_csr = sparse.csr_matrix(
        (csr_data, csr_indices, csr_indptr), shape=expected_csr.shape
    )
    np.testing.assert_array_equal(actual_csr.toarray(), expected_csr)
    assert not sys._is_gil_enabled()

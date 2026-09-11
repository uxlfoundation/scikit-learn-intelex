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

import importlib.util
import os
import subprocess
import sys
import sysconfig
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

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
    env = os.environ | {"PYTHON_GIL": "0"}

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


def test_daal4py_model_is_read_only_and_readable_concurrently():
    """Model wrappers hold a write-once native pointer.

    Readers dereference it without synchronization, so replacing it would be a
    use-after-free for a thread already inside a getter. Unpickling into a
    populated object is therefore rejected, and concurrent reads are safe.
    """
    import pickle

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

    with pytest.raises(ValueError, match="already-initialized"):
        model.__setstate__(state)

    # The supported path - unpickling allocates a fresh object - still works.
    # nosec B301: the input is pickle.dumps of an object created on the line
    # above, not untrusted data. Round-tripping is the behavior under test.
    assert pickle.loads(pickle.dumps(model)).NumberOfTrees == 4  # nosec

    start = Barrier(4)

    def read_state():
        start.wait()
        for _ in range(32):
            assert model.NumberOfTrees == 4
            assert model.__getstate__()
            assert repr(model)
            assert daal4py.getTreeState(model, 0, 2) is not None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(read_state) for _ in range(4)]
        for future in futures:
            future.result()

    assert not sys._is_gil_enabled()

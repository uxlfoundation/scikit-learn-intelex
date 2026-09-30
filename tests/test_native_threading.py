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

import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import numpy as np
import pytest

import daal4py


def test_model_is_read_only_and_readable_concurrently():
    """Model wrappers hold a write-once native pointer.

    Readers dereference it without synchronization, so replacing it would be a
    use-after-free for a thread already inside a getter. Unpickling into a
    populated object is therefore rejected, and concurrent reads are safe.
    """
    x = np.arange(400, dtype=np.float64).reshape(200, 2)
    y = (x[:, 0] > x[:, 1]).astype(np.int64).reshape(-1, 1)
    model = (
        daal4py.decision_forest_classification_training(nClasses=2, nTrees=4)
        .compute(x, y)
        .model
    )
    state = model.__getstate__()

    with pytest.raises(ValueError, match="already-initialized"):
        model.__setstate__(state)

    # The supported path - unpickling allocates a fresh object - still works.
    # nosec B301: the input is pickle.dumps of an object created above.
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


def test_shared_manager_does_not_invert_gil_and_native_mutex():
    """A waiter must not hold the GIL while the active call reattaches."""
    code = r"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import daal4py

x = np.arange(4000, dtype=np.float64).reshape(2000, 2)
y = (3.0 * x[:, 0] - 2.0 * x[:, 1] + 5.0).reshape(-1, 1)
training = daal4py.linear_regression_training()

def train(_):
    result = training.compute(x.copy(), y.copy())
    assert result.model is not None

with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(train, range(64)))
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=60)

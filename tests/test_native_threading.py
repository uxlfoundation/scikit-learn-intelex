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

import subprocess
import sys


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

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
from numpy.testing import assert_allclose


# TODO:
# add pytest params for checking different dataframe inputs/outputs.
def test_sklearnex_import_train_test_split():
    from sklearnex.model_selection import train_test_split

    X = np.arange(100).reshape((10, 10))
    y = np.arange(10)

    split = train_test_split(X, y, test_size=None, train_size=0.5)
    X_train, X_test, y_train, y_test = split
    assert len(y_test) == len(y_train)

    assert_allclose(X_train[:, 0], y_train * 10)
    assert_allclose(X_test[:, 0], y_test * 10)

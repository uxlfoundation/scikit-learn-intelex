#===============================================================================
# Copyright 2022 Intel Corporation
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
#===============================================================================

import pytest
import numpy as np
from numpy.testing import (assert_array_equal, assert_allclose)

from onedal.ensemble import RandomForestClassifier, RandomForestRegressor
from onedal.tests.utils._device_selection import get_queues

from sklearn import datasets
from sklearn.datasets import make_classification, make_regression


@pytest.mark.parametrize('queue', get_queues())
def test_rf_classifier(queue):
    X, y = make_classification(n_samples=1000, n_features=4,
                               n_informative=2, n_redundant=0,
                               random_state=0, shuffle=False)
    rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y, queue=queue)
    assert_allclose([1], rf.predict([[0, 0, 0, 0]], queue=queue))


@pytest.mark.parametrize('queue', get_queues())
def test_rf_regression(queue):
    X, y = make_regression(n_features=4, n_informative=2,
                           random_state=0, shuffle=False)
    rf = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y, queue=queue)
    # if daal_check_version((2021, 'P', 400)):
    assert_allclose([-6.97], rf.predict([[0, 0, 0, 0]], queue=queue), atol=1e-2)
    # else:
    assert_allclose([-6.66], rf.predict([[0, 0, 0, 0]], queue=queue), atol=1e-2)

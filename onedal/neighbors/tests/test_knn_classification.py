# ===============================================================================
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
# ===============================================================================

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn import datasets

from onedal.neighbors import KNeighborsClassifier
from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues())
def test_iris(queue):
    iris = datasets.load_iris()
    clf = KNeighborsClassifier(2).fit(iris.data, iris.target, queue=queue)
    assert clf.score(iris.data, iris.target, queue=queue) > 0.9
    assert_array_equal(clf.classes_, np.sort(clf.classes_))


@pytest.mark.parametrize("queue", get_queues())
def test_pickle(queue):
    if queue and queue.sycl_device.is_gpu:
        pytest.xfail("KNN classifier pickling for the GPU sycl_queue is buggy.")
    iris = datasets.load_iris()
    clf = KNeighborsClassifier(2).fit(iris.data, iris.target, queue=queue)
    expected = clf.predict(iris.data, queue=queue)

    import pickle

    dump = pickle.dumps(clf)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    result = clf2.predict(iris.data, queue=queue)
    assert_array_equal(expected, result)

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

# REFACTOR: Import from sklearnex instead of onedal
# Classification processing now happens in sklearnex layer
from sklearnex.neighbors import KNeighborsClassifier
from onedal.tests.utils._device_selection import get_queues


@pytest.mark.parametrize("queue", get_queues())
def test_iris(queue):
    import sys
    print(f"\n=== DEBUG test_iris START: queue={queue} ===", file=sys.stderr)
    # REFACTOR NOTE: queue parameter not used with sklearnex, but kept for test parametrization
    iris = datasets.load_iris()
    print(f"DEBUG test: iris.data type={type(iris.data)}, shape={iris.data.shape}", file=sys.stderr)
    print(f"DEBUG test: iris.target type={type(iris.target)}, shape={iris.target.shape}", file=sys.stderr)
    print(f"DEBUG test: Creating KNeighborsClassifier and calling fit", file=sys.stderr)
    clf = KNeighborsClassifier(2).fit(iris.data, iris.target)
    print(f"DEBUG test: fit completed, clf._fit_X type={type(getattr(clf, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
    print(f"DEBUG test: Calling score", file=sys.stderr)
    score = clf.score(iris.data, iris.target)
    print(f"DEBUG test: score completed, score={score}", file=sys.stderr)
    assert score > 0.9
    assert_array_equal(clf.classes_, np.sort(clf.classes_))
    print(f"=== DEBUG test_iris END ===\n", file=sys.stderr)


@pytest.mark.parametrize("queue", get_queues())
def test_pickle(queue):
    import sys
    print(f"\n=== DEBUG test_pickle START: queue={queue} ===", file=sys.stderr)
    # REFACTOR NOTE: queue parameter not used with sklearnex, but kept for test parametrization
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("KNN classifier pickling for the GPU sycl_queue is buggy.")
    iris = datasets.load_iris()
    print(f"DEBUG test: iris.data type={type(iris.data)}, shape={iris.data.shape}", file=sys.stderr)
    print(f"DEBUG test: iris.target type={type(iris.target)}, shape={iris.target.shape}", file=sys.stderr)
    print(f"DEBUG test: Creating KNeighborsClassifier and calling fit", file=sys.stderr)
    clf = KNeighborsClassifier(2).fit(iris.data, iris.target)
    print(f"DEBUG test: fit completed, clf._fit_X type={type(getattr(clf, '_fit_X', 'NOT_SET'))}", file=sys.stderr)
    print(f"DEBUG test: Calling predict", file=sys.stderr)
    expected = clf.predict(iris.data)
    print(f"DEBUG test: predict completed, expected type={type(expected)}, shape={expected.shape}", file=sys.stderr)

    import pickle

    print(f"DEBUG test: Pickling classifier", file=sys.stderr)
    dump = pickle.dumps(clf)
    print(f"DEBUG test: Unpickling classifier", file=sys.stderr)
    clf2 = pickle.loads(dump)

    assert type(clf2) == clf.__class__
    print(f"DEBUG test: Calling predict on unpickled classifier", file=sys.stderr)
    result = clf2.predict(iris.data)
    print(f"DEBUG test: predict completed, result type={type(result)}, shape={result.shape}", file=sys.stderr)
    assert_array_equal(expected, result)
    print(f"=== DEBUG test_pickle END ===\n", file=sys.stderr)
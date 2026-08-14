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

"""Reference-counting behaviour of the daal4py NumericTable conversions.

These exercise the ownership rules rather than numerical results: a table that
borrows a NumPy buffer must keep the owner alive, must not leak it, and the
``__2daalnt__`` protocol must consume the capsule it is handed exactly once.
"""

import gc
import sys

import numpy as np
import pytest

import daal4py._daal4py as backend


def _refcount(obj):
    gc.collect()
    # Subtract the temporary reference held by getrefcount's own argument.
    return sys.getrefcount(obj) - 1


def test_numeric_table_protocol_consumes_capsule_once():
    array = np.full((32, 4), 11.0, dtype=np.float64)

    class NumericTableProtocol:
        def __2daalnt__(self):
            return backend._make_nt_capsule_for_testing(array)

    obj = NumericTableProtocol()
    before = _refcount(array)
    for _ in range(64):
        np.testing.assert_array_equal(
            backend._roundtrip_nt_for_testing(obj),
            array,
        )
    assert _refcount(array) == before


def test_numeric_table_protocol_accepts_legacy_unnamed_capsule():
    array = np.full((16, 2), 5.0, dtype=np.float64)

    class LegacyProtocol:
        def __2daalnt__(self):
            return backend._make_nt_capsule_for_testing(array, legacy=True)

    np.testing.assert_array_equal(
        backend._roundtrip_nt_for_testing(LegacyProtocol()),
        array,
    )


def test_numeric_table_protocol_rejects_non_capsule():
    class BadProtocol:
        def __2daalnt__(self):
            return 42

    with pytest.raises(Exception):
        backend._roundtrip_nt_for_testing(BadProtocol())


def test_numeric_table_protocol_is_not_cached_across_objects():
    """Regression test: the result used to be held in a function-level static.

    The static was written on every call but returned the same shared pointer
    object, so two different inputs could not be converted independently.
    """
    first = np.full((8, 2), 1.0, dtype=np.float64)
    second = np.full((8, 2), 2.0, dtype=np.float64)

    class Protocol:
        def __init__(self, array):
            self.array = array

        def __2daalnt__(self):
            return backend._make_nt_capsule_for_testing(self.array)

    for _ in range(4):
        np.testing.assert_array_equal(
            backend._roundtrip_nt_for_testing(Protocol(first)), first
        )
        np.testing.assert_array_equal(
            backend._roundtrip_nt_for_testing(Protocol(second)), second
        )


@pytest.mark.parametrize(
    "array",
    [
        # C-contiguous: converted to a HomogenNumericTable in place.
        np.arange(64, dtype=np.float64).reshape(16, 4),
        # F-contiguous: converted column-wise to an SOA table.
        np.asfortranarray(np.arange(64, dtype=np.float64).reshape(16, 4)),
        # Non-contiguous: wrapped by NpyNumericTable.
        np.arange(128, dtype=np.float64).reshape(16, 8)[:, ::2],
    ],
    ids=["c_contiguous", "f_contiguous", "non_contiguous"],
)
def test_table_conversion_balances_input_references(array):
    before = _refcount(array)
    for _ in range(16):
        backend._roundtrip_nt_for_testing(array)
    assert _refcount(array) == before


def test_table_keeps_numpy_owner_alive_after_input_is_dropped():
    expected = np.arange(64, dtype=np.float64).reshape(16, 4)
    array = expected.copy()
    result = backend._roundtrip_nt_for_testing(array)
    del array
    gc.collect()
    # Churn the allocator so a freed buffer would be reused.
    _ = [np.empty((16, 4), dtype=np.float64) for _ in range(128)]
    np.testing.assert_array_equal(result, expected)

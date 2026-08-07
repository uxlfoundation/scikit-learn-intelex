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

"""Tests for the shared result-option tokenizer in onedal/common/result_options.hpp.

The tokenizer is a header-only helper with no Python binding of its own, but
every estimator that accepts a ``result_option`` string feeds it through, so it
is exercised here through that public path rather than through a wrapper added
solely for testing.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from onedal.basic_statistics import BasicStatistics
from onedal.tests.utils._device_selection import get_queues


def _compute(options, queue=None):
    data = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)
    estimator = BasicStatistics(result_options=options)
    return estimator.fit(data, queue=queue)


@pytest.mark.parametrize("queue", get_queues())
def test_multiple_options_are_all_parsed(queue):
    """Every token in a separator-joined string must reach the backend."""
    result = _compute(["min", "max", "mean"], queue=queue)

    np.testing.assert_allclose(np.asarray(result.min_), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(result.max_), [4.0, 5.0])
    np.testing.assert_allclose(np.asarray(result.mean_), [2.0, 3.0])


@pytest.mark.parametrize("queue", get_queues())
def test_single_option_is_parsed(queue):
    """A string with no separator is one token, not zero."""
    result = _compute(["mean"], queue=queue)
    np.testing.assert_allclose(np.asarray(result.mean_), [2.0, 3.0])


@pytest.mark.parametrize("queue", get_queues())
def test_repeated_option_is_idempotent(queue):
    """Repeating a token must not change the result; the backend ORs the flags."""
    once = _compute(["mean"], queue=queue)
    twice = _compute(["mean", "mean"], queue=queue)
    np.testing.assert_allclose(np.asarray(once.mean_), np.asarray(twice.mean_))


@pytest.mark.parametrize(
    "options",
    [
        ["not_a_real_option"],
        ["mean", "not_a_real_option"],
        # A token that is a prefix of a real one must not match it.
        ["mea"],
        # ``_`` is a word character, so this is one unknown token rather than two
        # known ones.
        ["min_max"],
    ],
)
@pytest.mark.parametrize("queue", get_queues())
def test_unrecognized_option_is_rejected(options, queue):
    """The callback throws for unknown tokens and that must surface to Python.

    ONEDAL_PARAM_DISPATCH_THROW_INVALID_VALUE raises std::runtime_error, which
    pybind11 translates to RuntimeError.
    """
    with pytest.raises(RuntimeError, match="Invalid value for parameter"):
        _compute(options, queue=queue)


@pytest.mark.parametrize("separator", ["|", ",", " ", "\0", "\xff", "é"])
@pytest.mark.parametrize("queue", get_queues())
def test_any_non_word_byte_separates_tokens(separator, queue):
    """Token boundaries are every byte outside [A-Za-z0-9_], matching ECMAScript \\w.

    This covers the cases the standalone native test used to assert - embedded
    NULs and bytes above 0x7f - through the public path. ``min`` and ``max`` are
    both real options, so a correctly split string yields both results.
    """
    result = _compute([f"min{separator}max"], queue=queue)

    np.testing.assert_allclose(np.asarray(result.min_), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(result.max_), [4.0, 5.0])


def test_concurrent_parsing_is_consistent():
    """The tokenizer holds no shared state, so concurrent callers must not interfere.

    This is the property the native test asserted; it is kept here because it is
    the reason the shared helper replaced a per-call std::regex.
    """
    expected = np.asarray(_compute(["min", "max", "mean"]).mean_)

    def parse_and_compute(_):
        return np.asarray(_compute(["min", "max", "mean"]).mean_)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(parse_and_compute, range(64)))

    for result in results:
        np.testing.assert_allclose(result, expected)

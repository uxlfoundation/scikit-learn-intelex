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
solely for testing. That also proves the parsed tokens actually arrive as the
requested results, not merely that a string was split.

The one exception is the separator test, which has to read the raw backend
result: ``fit()`` can only unpack options that were joined with ``"|"``. See
``_compute_raw`` below.

Testing through the estimators rather than through a standalone native binary is
deliberate. It needs no extra build artifact or CI step, and it proves more: that
the parsed tokens arrive as the requested results, not merely that a string was
split into the expected pieces.
"""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from onedal.basic_statistics import BasicStatistics
from onedal.datatypes import from_table, to_table
from onedal.tests.utils._device_selection import get_queues
from onedal.utils import _sycl_queue_manager as QM


def _make_data():
    # A fresh array per call: oneDAL may modify input arrays in place, and the
    # concurrency test below runs several computations at once.
    return np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float64)


def _compute(options, queue=None):
    """Drive the tokenizer through the full public path, including fit()."""
    estimator = BasicStatistics(result_options=options)
    return estimator.fit(_make_data(), queue=queue)


def _compute_raw(options, fields, queue=None):
    """Drive the tokenizer through the backend, bypassing fit()'s result unpacking.

    fit() maps results back onto Python attributes by iterating ``self.options``
    and doing ``getattr(result, option)``, so it can only ever see options that
    were joined with ``"|"`` - the separator it splits on, and the only one it
    can round-trip. Testing that *any* non-word byte separates tokens therefore
    has to read the raw backend result, whose fields are named after the tokens
    the C++ side actually parsed. The conversion below is otherwise identical to
    what fit() does.
    """
    data = _make_data()
    estimator = BasicStatistics(result_options=options)
    with QM.manage_global_queue(queue, data) as global_queue:
        data_table, weights_table = to_table(data, None, queue=global_queue)
        result = estimator._compute_raw(data_table, weights_table, data_table.dtype)
        # 2D table [1, n], as in fit().
        return [from_table(getattr(result, field), like=data)[0, :] for field in fields]


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

    Embedded NULs and bytes above 0x7f are covered here, which is where the
    hand-written scan differs most visibly from the ``std::regex`` it replaced.
    ``min`` and ``max`` are both real options, so a correctly split string
    yields both results.

    Note the ``é`` case is a multi-byte UTF-8 character, so this also pins down
    that a token is split on its lead byte rather than mangled.
    """
    minimum, maximum = _compute_raw([f"min{separator}max"], ["min", "max"], queue=queue)

    np.testing.assert_allclose(np.asarray(minimum), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(maximum), [4.0, 5.0])


def test_concurrent_parsing_is_consistent():
    """The tokenizer holds no shared state, so concurrent callers must not interfere.

    This is the property that motivated the change: the ``std::regex`` it
    replaced was imbued with the global locale, so the meaning of ``\\w``
    depended on process-wide state another thread could mutate.
    """
    expected = np.asarray(_compute(["min", "max", "mean"]).mean_)

    def parse_and_compute(_):
        return np.asarray(_compute(["min", "max", "mean"]).mean_)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(parse_and_compute, range(64)))

    for result in results:
        np.testing.assert_allclose(result, expected)

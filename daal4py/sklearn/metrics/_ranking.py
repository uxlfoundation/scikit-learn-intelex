# ==============================================================================
# Copyright 2020 Intel Corporation
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

# ==============================================================================
# BSD 3-Clause License
#
# Copyright (c) 2007-2026 The scikit-learn developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================

import logging
from collections.abc import Sequence
from functools import partial

import numpy as np
from scipy import sparse as sp
from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
from sklearn.metrics._base import _average_binary_score
from sklearn.metrics._ranking import _binary_roc_auc_score
from sklearn.metrics._ranking import _multiclass_roc_auc_score as multiclass_roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_array
from sklearn.utils.multiclass import is_multilabel

import daal4py as d4p

from .._utils import (
    PatchingConditionsChain,
    check_is_array_api,
    get_patch_message,
    sklearn_check_version,
)
from ..utils.validation import _assert_all_finite

if sklearn_check_version("1.3"):
    from sklearn.utils._param_validation import (
        Interval,
        Real,
        StrOptions,
        validate_params,
    )

try:
    import pandas as pd

    pandas_is_imported = True
except ImportError:
    pandas_is_imported = False


def _daal_type_of_target(y):
    valid = (
        isinstance(y, Sequence) or sp.isspmatrix(y) or hasattr(y, "__array__")
    ) and not isinstance(y, str)

    if not valid:
        raise ValueError(
            "Expected array-like (array or non-string sequence), " "got %r" % y
        )

    sparse_pandas = y.__class__.__name__ in ["SparseSeries", "SparseArray"]
    if sparse_pandas:
        raise ValueError("y cannot be class 'SparseSeries' or 'SparseArray'")

    if is_multilabel(y):
        return "multilabel-indicator"

    try:
        y = np.asarray(y)
    except ValueError:
        # Known to fail in numpy 1.3 for array of arrays
        return "unknown"

    # The old sequence of sequences format
    try:
        if (
            not hasattr(y[0], "__array__")
            and isinstance(y[0], Sequence)
            and not isinstance(y[0], str)
        ):
            raise ValueError(
                "You appear to be using a legacy multi-label data"
                " representation. Sequence of sequences are no"
                " longer supported; use a binary array or sparse"
                " matrix instead - the MultiLabelBinarizer"
                " transformer can convert to this format."
            )
    except IndexError:
        pass

    # Invalid inputs
    if y.ndim > 2 or (
        y.dtype == object and len(y) != 0 and not isinstance(y.flat[0], str)
    ):
        return "unknown"  # [[[1, 2]]] or [obj_1] and not ["label_1"]

    if y.ndim == 2 and y.shape[1] == 0:
        return "unknown"  # [[]]

    if y.ndim == 2 and y.shape[1] > 1:
        suffix = "-multioutput"  # [[1, 2], [1, 2]]
    else:
        suffix = ""  # [1, 2, 3] or [[1], [2], [3]]

    # check float and contains non-integer float values
    if y.dtype.kind == "f" and np.any(y != y.astype(int)):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        _assert_all_finite(y)
        return "continuous" + suffix

    unique = np.sort(pd.unique(y.ravel())) if pandas_is_imported else np.unique(y)

    if (len(unique) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
        result = ("multiclass" + suffix, None)
    else:
        result = ("binary", unique)  # [1, 2] or [["a"], ["b"]]
    return result


def roc_auc_score(
    y_true,
    y_score,
    *,
    average="macro",
    sample_weight=None,
    max_fpr=None,
    multi_class="raise",
    labels=None,
):
    _patching_status = PatchingConditionsChain("sklearn.metrics.roc_auc_score")
    _dal_ready = _patching_status.and_conditions(
        [
            (sample_weight is None, "Sample weights are not supported"),
            (max_fpr is None, "'max_fpr' is not supported"),
            (
                not (
                    check_is_array_api(y_true)
                    or check_is_array_api(y_score)
                    or check_is_array_api(sample_weight)
                    or check_is_array_api(labels)
                ),
                "Array API inputs other than NumPy are not supported.",
            ),
        ]
    )
    if not _dal_ready:
        _patching_status.write_log()
        return _sklearn_roc_auc_score(
            y_true,
            y_score,
            average=average,
            sample_weight=sample_weight,
            max_fpr=max_fpr,
            multi_class=multi_class,
            labels=labels,
        )

    y_type = _daal_type_of_target(y_true)
    y_true = check_array(y_true, ensure_2d=False, dtype=None)
    y_score = check_array(y_score, ensure_2d=False)

    _dal_ready = _patching_status.and_conditions(
        [
            (
                y_type[0] == "binary"
                and not (y_score.ndim == 2 and y_score.shape[1] > 2),
                "y_true type is not one-dimensional binary.",
            )
        ]
    )
    _patching_status.write_log()
    if not _dal_ready:
        return _sklearn_roc_auc_score(
            y_true,
            y_score,
            average=average,
            sample_weight=sample_weight,
            max_fpr=max_fpr,
            multi_class=multi_class,
            labels=labels,
        )

    # Comment 2026-03-03: the original code here was a copy-paste from an older scikit-learn
    # version with some parts replaced with calls to oneDAL. The logic was then modified to
    # fall back to scikit-learn's 'roc_auc_score' directly and as early as possible, so some
    # code branches below this comment might be unreachable. Note that the logic of not falling
    # back has the advantage of avoiding two calls to 'check_array', but after some input
    # has already been processed by it, passing it again to that function shouldn't do any
    # data conversions.

    if y_type[0] == "multiclass" or (
        y_type[0] == "binary" and y_score.ndim == 2 and y_score.shape[1] > 2
    ):
        # do not support partial ROC computation for multiclass
        if max_fpr is not None and max_fpr != 1.0:
            raise ValueError(
                "Partial AUC computation not available in "
                "multiclass setting, 'max_fpr' must be"
                " set to `None`, received `max_fpr={0}` "
                "instead".format(max_fpr)
            )
        if multi_class == "raise":
            raise ValueError("multi_class must be in ('ovo', 'ovr')")

        return multiclass_roc_auc_score(
            y_true, y_score, labels, multi_class, average, sample_weight
        )

    if y_type[0] == "binary":
        labels = y_type[1]
        _dal_ready = _patching_status.and_conditions(
            [
                (len(labels) == 2, "Number of unique labels is not equal to 2."),
                (max_fpr is None, "Maximum false-positive rate is not supported."),
                (sample_weight is None, "Sample weights are not supported."),
            ]
        )
        if _dal_ready:
            if not np.array_equal(labels, [0, 1]) or labels.dtype == bool:
                y_true = label_binarize(y_true, classes=labels)[:, 0]
                if hasattr(y_score, "dtype") and y_score.dtype == bool:
                    y_score = label_binarize(y_score, classes=labels)[:, 0]
            result = d4p.daal_roc_auc_score(y_true.reshape(-1, 1), y_score.reshape(-1, 1))
            if result != -1:
                return result
            logging.info(
                "sklearn.metrics.roc_auc_score: "
                + get_patch_message("sklearn_after_daal")
            )
        # return to sklearn implementation
        y_true = label_binarize(y_true, classes=labels)[:, 0]

    return _average_binary_score(
        partial(_binary_roc_auc_score, max_fpr=max_fpr),
        y_true,
        y_score,
        average,
        sample_weight=sample_weight,
    )


if sklearn_check_version("1.3"):
    roc_auc_score = validate_params(
        {
            "y_true": ["array-like"],
            "y_score": ["array-like"],
            "average": [StrOptions({"micro", "macro", "samples", "weighted"}), None],
            "sample_weight": ["array-like", None],
            "max_fpr": [Interval(Real, 0.0, 1, closed="right"), None],
            "multi_class": [StrOptions({"raise", "ovr", "ovo"})],
            "labels": ["array-like", None],
        },
        prefer_skip_nested_validation=True,
    )(roc_auc_score)

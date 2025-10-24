# ==============================================================================
# Copyright contributors to the oneDAL Project
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

from sklearn.preprocessing import LabelEncoder as _sklearn_LabelEncoder

from daal4py.sklearn._utils import sklearn_check_version

from ._array_api import get_namespace
from .validation import _check_sample_weight

if not sklearn_check_version("1.7"):
    from sklearn.utils.class_weight import (
        compute_class_weight as _sklearn_compute_class_weight,
    )

    def compute_class_weight(class_weight, *, classes, y, sample_weight=None):
        return _sklearn_compute_class_weight(class_weight, classes=classes, y=y)

else:
    from sklearn.utils.class_weight import compute_class_weight


def _compute_class_weight(class_weight, *, classes, y, sample_weight=None):
    # this duplicates sklearn code in order to enable it for array API.
    # Note for the use of LabelEncoder this is only valid for sklearn
    # versions >= 1.6.
    xp, is_array_api_compliant = get_namespace(classes, y, sample_weight)

    if not is_array_api_compliant:
        # use the sklearn version for standard use.
        return compute_class_weight(class_weight, classes=classes, y=y, sample_weight=sample_weight)

    sety = xp.unique_values(y)
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = xp.ones(
            (classes.shape[0],), dtype=xp.float64, device=getattr(classes, "device", None)
        )
    elif class_weight == "balanced":
        if not sklearn_check_version("1.6"):
            raise RuntimeError(
                "array API support with 'balanced' keyword not supported for sklearn <1.6"
            )
        # Find the weight of each class as present in y.
        le = _sklearn_LabelEncoder()
        y_ind = le.fit_transform(y)
        if not all([item in le.classes_ for item in classes]):
            raise ValueError("classes should have valid labels that are in y")

        sample_weight = _check_sample_weight(sample_weight, y)
        # scikit-learn implementation uses numpy.bincount, which does a combined
        # min and max search, only erroring when a value < 0. Replicating this
        # exactly via array API would cause another O(n) evaluation (by doing
        # min and max separately). However this check can be removed due to the
        # nature of the LabelEncoder. Therefore only the maximum is found, and
        # then core logic of bincount is replicated:
        # https://github.com/numpy/numpy/blob/main/numpy/_core/src/multiarray/compiled_base.c
        weighted_class_counts = xp.zeros(
            (xp.max(y_ind) + 1,),
            dtype=sample_weight.dtype,
            device=getattr(y, "device", None),
        )

        # use a more GPU-friendly summation approach for collecting weighted_class_counts
        for w_idx in range(weighted_class_counts.shape[0]):
            weighted_class_counts[w_idx] = xp.sum(sample_weight[y_ind == w_idx])

        recip_freq = xp.sum(weighted_class_counts) / (
            le.classes_.shape[0] * weighted_class_counts
        )

        weight = xp.take(recip_freq, le.transform(classes))
    else:
        # user-defined dictionary
        weight = xp.ones(
            (classes.shape[0],), dtype=xp.float64, device=getattr(classes, "device", None)
        )
        unweighted_classes = []
        for i, c in enumerate(classes):
            if (fc := float(c)) in class_weight:
                # array API has only numeric datatypes, convert to float for generality
                # complex values should never be observed by this function
                weight[i] = class_weight[fc]
            else:
                unweighted_classes.append(c)

        n_weighted_classes = classes.shape[0] - len(unweighted_classes)
        if unweighted_classes and n_weighted_classes != len(class_weight):
            raise ValueError(
                f"The classes, {unweighted_classes}, are not in" " class_weight"
            )

    return weight

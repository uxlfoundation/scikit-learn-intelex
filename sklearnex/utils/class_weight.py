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

if sklearn_check_version("1.9"):
    from sklearn.utils._array_api import get_namespace_and_device, move_to

if not sklearn_check_version("1.7"):
    from sklearn.utils.class_weight import (
        compute_class_weight as _sklearn_compute_class_weight,
    )

    def compute_class_weight(class_weight, *, classes, y, sample_weight=None):
        return _sklearn_compute_class_weight(class_weight, classes=classes, y=y)

else:
    from sklearn.utils.class_weight import compute_class_weight


# Note: last argument 'y_encoded' is not present in scikit-learn's version.
# It is added in order to avoid recoding 'y' into integers more than once
# when this function is called after that step has already happened.
def _compute_class_weight(
    class_weight, *, classes, y, sample_weight=None, y_encoded=None
):
    # this duplicates sklearn code in order to enable it for array API.
    # Note for the use of LabelEncoder this is only valid for sklearn
    # versions >= 1.6.
    # Comment 2026-04-21: this function has been adjusted to follow the
    # 'everything follows X' logic of scikit-learn's array API. By this point,
    # scikit-learn already supports array API for this function, but theirs
    # might be inefficient since it moves the data to NumPy internally, so
    # it's no longer a duplicate of scikit-learn's.

    if sklearn_check_version("1.9"):
        xp_y, is_array_api_compliant, device = get_namespace_and_device(y)
    else:
        xp, is_array_api_compliant = get_namespace(classes, y, sample_weight)

    if not is_array_api_compliant:
        # use the sklearn version for standard use.
        return compute_class_weight(
            class_weight, classes=classes, y=y, sample_weight=sample_weight
        )

    if sklearn_check_version("1.9"):
        if sample_weight is not None:
            xp, _, device = get_namespace_and_device(sample_weight)
            if y_encoded is not None:
                y_encoded = move_to(y_encoded, xp=xp, device=device)
        elif y_encoded is not None:
            xp, _, device = get_namespace_and_device(y_encoded)
        else:
            xp = xp_y

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
        if y_encoded is None:
            le = _sklearn_LabelEncoder()
            y_ind = le.fit_transform(y)
            if not all([item in le.classes_ for item in classes]):
                raise ValueError("classes should have valid labels that are in y")

            if sklearn_check_version("1.9"):
                y_ind = move_to(y_ind, xp=xp, device=device)

            n_classes = le.classes_.shape[0]
        else:
            y_ind = y_encoded
            n_classes = classes.shape[0]

        sample_weight = _check_sample_weight(sample_weight, y_ind)
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
            device=(
                getattr(y_ind, "device", None)
                if not sklearn_check_version("1.9")
                else device
            ),
        )

        # use a more GPU-friendly summation approach for collecting weighted_class_counts
        for w_idx in range(weighted_class_counts.shape[0]):
            weighted_class_counts[w_idx] = xp.sum(sample_weight[y_ind == w_idx])

        recip_freq = xp.sum(weighted_class_counts) / (n_classes * weighted_class_counts)

        if y_encoded is None:
            ind = le.transform(classes)
            if sklearn_check_version("1.9"):
                ind = move_to(ind, xp=xp, device=device)
            weight = xp.take(recip_freq, ind)
        else:
            weight = recip_freq
    else:
        # user-defined dictionary
        weight = xp.ones(
            (classes.shape[0],),
            dtype=xp.float64,
            device=(
                getattr(classes, "device", None)
                if not sklearn_check_version("1.9")
                else device
            ),
        )
        unweighted_classes = []
        for i, c in enumerate(classes):
            if (fc := float(c)) in class_weight:
                weight[i] = class_weight[fc]
            else:
                unweighted_classes.append(c)

        n_weighted_classes = classes.shape[0] - len(unweighted_classes)
        if unweighted_classes and n_weighted_classes != len(class_weight):
            raise ValueError(
                f"The classes, {unweighted_classes}, are not in" " class_weight"
            )

    return weight

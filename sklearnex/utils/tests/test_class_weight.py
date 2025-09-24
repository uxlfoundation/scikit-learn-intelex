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

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.utils import compute_class_weight

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.utils.class_weight import _compute_class_weight


@pytest.mark.skipif(not sklearn_check_version("1.6"), reason="lacks array API support")
@pytest.mark.parametrize("class_weight", [None, "balanced", "ramp"])
@pytest.mark.parametrize(
    "dataframe,queue", get_dataframes_and_queues("array_api_strict,dpctl")
)
def test_compute_class_weight_array_api(class_weight, dataframe, queue):
    # This verifies that array_api functionality matches sklearn

    _, y = load_iris(return_X_y=True)
    classes = np.unique()

    y_xp = _convert_to_dataframe(y, target_df=dataframe, device=queue)
    classes_xp = _convert_to_dataframe(classes, target_df=dataframe, device=queue)

    sample_weight = (
        np.ones(y.shape, dtype=np.float64) if class_weight == "balanced" else None
    )

    if class_weight == "ramp":
        class_weight = {int(i): int(i) for i in np.unique(y)}

    weight_np = compute_class_weight(
        class_weight, classes, y, sample_weight=sample_weight
    )

    if sample_weight:
        sample_weight = _convert_to_dataframe(
            sample_weight, target_df=dataframe, device=queue
        )

    # evaluate custom sklearnex array API functionality
    with config_context(array_api_dispatch=True):
        weight_xp = _compute_class_weight(
            class_weight, classes_xp, y_xp, sample_weight=sample_weight
        )

    np.testing.assert_allclose(_as_numpy(weight_xp), weight_np)

# ==============================================================================
# Copyright 2024 Intel Corporation
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
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)

# TODO:
# add test suit for dpctl.tensor, dpnp.ndarray, numpy.ndarray without config_context(array_api_dispatch=True)).
# TODO:
# extend for DPNP inputs.


@pytest.mark.skipif(
    not sklearn_check_version("1.4"),
    reason="Array API dispatch requires sklearn 1.4 version",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(
        dataframe_filter_="numpy,dpctl,array_api", device_filter_="cpu,gpu"
    ),
)
def test_get_namespace_with_config_context(dataframe, queue):
    """Test `get_namespace` with `array_api_dispatch` enabled."""
    from sklearnex import config_context
    from sklearnex.utils._array_api import get_namespace

    array_api_compat = pytest.importorskip("array_api_compat")

    X_np = np.asarray([[1, 2, 3]])
    X = _convert_to_dataframe(X_np, sycl_queue=queue, target_df=dataframe)

    with config_context(array_api_dispatch=True):
        xp_out, is_array_api_compliant = get_namespace(X)
        assert is_array_api_compliant
        if not dataframe in "numpy,array_api":
            # Rather than array_api_compat.get_namespace raw output
            # `get_namespace` has specific wrapper classes for `numpy.ndarray`
            # or `array-api-strict`.
            assert xp_out == array_api_compat.get_namespace(X)

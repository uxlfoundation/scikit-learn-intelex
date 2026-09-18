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

import numpy as np
import pytest

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex import config_context
from sklearnex.tests.utils.spmd import (
    _get_local_tensor,
    _mpi_libs_and_gpu_available,
    _spmd_assert_allclose,
)

pytestmark = pytest.mark.skipif(
    not daal_check_version((2026, "P", 200)),
    reason="HDBSCAN requires oneDAL >= 2026.2",
)


@pytest.mark.skipif(
    not _mpi_libs_and_gpu_available,
    reason="GPU device and MPI libs required for test",
)
@pytest.mark.parametrize(
    "dataframe,queue",
    get_dataframes_and_queues(dataframe_filter_="dpnp,torch", device_filter_="gpu"),
)
@pytest.mark.mpi
def test_hdbscan_spmd_gold(dataframe, queue):
    # Import spmd and batch algo
    from sklearnex.preview.cluster import HDBSCAN as HDBSCAN_Batch
    from sklearnex.spmd.cluster import HDBSCAN as HDBSCAN_SPMD

    # three tight groups of five samples, far apart from each other
    data = np.array(
        [[1, 2], [2, 2], [2, 3], [1, 3], [2, 1]]
        + [[28, 27], [28, 28], [29, 28], [27, 28], [28, 29]]
        + [[55, 80], [55, 81], [56, 80], [54, 80], [55, 79]],
        dtype=np.float64,
    )

    local_dpt_data = _convert_to_dataframe(
        _get_local_tensor(data), sycl_queue=queue, target_df=dataframe
    )

    # Ensure labels from fit of batch algo matches spmd
    with config_context(array_api_dispatch=True):
        spmd_model = HDBSCAN_SPMD(min_cluster_size=5).fit(local_dpt_data)
    batch_model = HDBSCAN_Batch(min_cluster_size=5).fit(data)

    _spmd_assert_allclose(spmd_model.labels_, batch_model.labels_)

    # Ensure meaningful test setup
    assert len(np.unique(batch_model.labels_)) == 3

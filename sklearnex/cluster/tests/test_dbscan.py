# ===============================================================================
# Copyright 2021 Intel Corporation
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
from numpy.testing import assert_allclose

from daal4py.sklearn._utils import sklearn_check_version
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    get_dataframes_and_queues,
)


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_dbscan(dataframe, queue):
    from sklearnex.cluster import DBSCAN

    X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    dbscan = DBSCAN(eps=3, min_samples=2).fit(X)
    assert "sklearnex" in dbscan.__module__

    result = _as_numpy(dbscan.labels_)
    expected = np.array([0, 0, 0, 1, 1, -1], dtype=np.int32)
    assert_allclose(expected, result)


if sklearn_check_version("1.9"):
    import array_api_strict
    from sklearn.datasets import make_blobs
    from sklearn.utils._array_api import (
        get_namespace_and_device,
        move_estimator_to,
        move_to,
    )

    from daal4py.sklearn._utils import _package_check_version
    from onedal.tests.utils._dataframes_support import (
        _as_numpy,
        dpnp_available,
        torch_available,
        torch_xpu_available,
    )
    from onedal.tests.utils._device_selection import (
        is_sycl_device_available,
    )
    from sklearnex.tests.utils.misc import assert_same_namespace

    if dpnp_available:
        import dpnp
    if torch_available:
        import torch

    @pytest.mark.skipif(
        not sklearn_check_version("1.9"),
        reason="Functionality introduced in later sklearn versions",
    )
    @pytest.mark.skipif(
        not _package_check_version("2.2", np.__version__),
        reason="Requires more recent NumPy version",
    )
    @pytest.mark.parametrize(
        "array_input_like",
        [np.arange(1), array_api_strict.arange(1)]
        + (
            [dpnp.arange(1, device="gpu")]
            if dpnp_available and is_sycl_device_available("gpu")
            else []
        )
        # Note: 'move_to' has issues with Torch inputs
        # in older sklearn versions.
        + (
            [torch.arange(1, device="xpu")]
            if torch_available and torch_xpu_available and sklearn_check_version("1.10")
            else []
        ),
    )
    @pytest.mark.parametrize(
        "array_move_like",
        [np.arange(1), array_api_strict.arange(1)]
        + (
            [dpnp.arange(1, device="gpu")]
            if dpnp_available and is_sycl_device_available("gpu")
            else []
        )
        + (
            [torch.arange(1, device="xpu")]
            if torch_available and torch_xpu_available and sklearn_check_version("1.10")
            else []
        ),
    )
    def test_dbscan_move_estimator_to(array_input_like, array_move_like, with_array_api):
        from sklearnex.cluster import DBSCAN

        X, _ = make_blobs(n_samples=20, random_state=123)

        xp_in, _, device_in = get_namespace_and_device(array_input_like)
        X_in = move_to(X, xp=xp_in, device=device_in)

        model = DBSCAN().fit(X_in)

        xp_move, _, device_move = get_namespace_and_device(array_move_like)
        model_moved = move_estimator_to(model, xp=xp_move, device=device_move)

        # TODO: update this once scikit-learn introduces a config option
        # to control whether the attributes are always numpy or follow 'X'
        for attr in ["core_sample_indices_", "components_", "labels_"]:
            assert_same_namespace(getattr(model, attr), array_input_like)
            assert_same_namespace(getattr(model_moved, attr), array_move_like)

        attrs_orig = dir(model)
        attrs_moved = dir(model_moved)
        for attr in attrs_orig:
            assert attr in attrs_moved

# ===============================================================================
# Copyright 2023 Intel Corporation
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

from daal4py.sklearn._utils import daal_check_version
from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from sklearnex.preview.covariance import EmpiricalCovariance


@pytest.fixture
def hyperparameters(request):
    hparams = EmpiricalCovariance.get_hyperparameters("fit")

    def restore_hyperparameters():
        EmpiricalCovariance.reset_hyperparameters("fit")

    request.addfinalizer(restore_hyperparameters)
    return hparams


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("macro_block", [None, 2])
@pytest.mark.parametrize("grain_size", [None, 2])
@pytest.mark.parametrize("assume_centered", [True, False])
def test_sklearnex_import_covariance(
    hyperparameters, dataframe, queue, macro_block, grain_size, assume_centered
):
    X = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    empcov = EmpiricalCovariance(assume_centered=assume_centered)

    if daal_check_version((2025, "P", 700)):
        if macro_block is not None:
            if queue and queue.sycl_device.is_gpu:
                pytest.skip("Test for CPU-only functionality")
            hyperparameters.cpu_macro_block = macro_block
        if grain_size is not None:
            if queue and queue.sycl_device.is_gpu:
                pytest.skip("Test for CPU-only functionality")
            hyperparameters.cpu_grain_size = grain_size

    result = empcov.fit(X)

    expected_covariance = np.array([[0, 0], [0, 0]])
    expected_means = np.array([0, 0])

    if assume_centered:
        expected_covariance = np.array([[0, 0], [0, 1]])
    else:
        expected_means = np.array([0, 1])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)

    X = np.array([[1, 2], [3, 6]])

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    result = empcov.fit(X)

    if assume_centered:
        expected_covariance = np.array([[5, 10], [10, 20]])
    else:
        expected_covariance = np.array([[1, 2], [2, 4]])
        expected_means = np.array([2, 4])

    assert_allclose(expected_covariance, result.covariance_)
    assert_allclose(expected_means, result.location_)


@pytest.mark.skipif(
    not daal_check_version((2025, "P", 700)),
    reason="Functionality introduced in a later version",
)
@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("assume_centered", [True, False])
def test_non_batched_covariance(hyperparameters, dataframe, queue, assume_centered):
    if queue and queue.sycl_device.is_gpu:
        pytest.skip("Test for CPU-only functionality")

    from sklearnex.preview.covariance import EmpiricalCovariance

    X = np.random.default_rng(seed=123).random(size=(20, 5))

    hyperparameters.cpu_max_cols_batched = np.iinfo(np.int32).max
    res_batched = EmpiricalCovariance(assume_centered=assume_centered).fit(X).covariance_

    hyperparameters.cpu_max_cols_batched = 1
    res_non_batched = (
        EmpiricalCovariance(assume_centered=assume_centered).fit(X).covariance_
    )

    np.testing.assert_allclose(res_non_batched, res_batched)

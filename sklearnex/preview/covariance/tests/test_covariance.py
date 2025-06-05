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


# Note: this is arranged as a fixture with a finalizer instead of as a parameter
# 'True' / 'False' in order to undo the changes later so that it doesn't affect
# other tests afterwards. It returns a function instead of making the change
# directly, in order to avoid importing the estimator class before the import test
# itself, but it still needs to import the class inside the the function that it
# returns due to serialization logic in pytest causing differences w.r.t. current
# closure where the function is called.
@pytest.fixture(params=[False, True])
def hyperparameters_route(request):
    def change_parameters(queue, macro_block, grain_size):
        from sklearnex.preview.covariance import EmpiricalCovariance

        if request.param and daal_check_version((2025, "P", 700)):
            if queue and queue.sycl_device.is_gpu:
                pytest.skip("Test for CPU-only functionality")

            hparams = EmpiricalCovariance.get_hyperparameters("fit")
            if macro_block is not None:
                hyperparameters_route.curr_cpu_macro_block = hparams.cpu_macro_block
                hparams.cpu_macro_block = macro_block
            else:
                hyperparameters_route.curr_cpu_macro_block = None
            if grain_size is not None:
                hyperparameters_route.curr_cpu_grain_size = hparams.cpu_grain_size
                hparams.cpu_grain_size = grain_size
            else:
                hyperparameters_route.curr_cpu_grain_size = None
        elif request.param and not daal_check_version((2025, "P", 700)):
            pytest.skip("Functionality introduced in later versions")

    def restore_params():
        from sklearnex.preview.covariance import EmpiricalCovariance

        if request.param and daal_check_version((2025, "P", 500)):
            hparams = EmpiricalCovariance.get_hyperparameters("fit")
            if (
                hasattr(hyperparameters_route, "curr_cpu_macro_block")
                and hyperparameters_route.curr_cpu_macro_block is not None
            ):
                hparams.cpu_macro_block = hyperparameters_route.curr_cpu_macro_block
            if (
                hasattr(hyperparameters_route, "curr_cpu_grain_size")
                and hyperparameters_route.curr_cpu_grain_size is not None
            ):
                hparams.cpu_grain_size = hyperparameters_route.curr_cpu_grain_size

    request.addfinalizer(restore_params)
    return change_parameters


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
@pytest.mark.parametrize("macro_block", [None, 2])
@pytest.mark.parametrize("grain_size", [None, 2])
@pytest.mark.parametrize("assume_centered", [True, False])
def test_sklearnex_import_covariance(
    dataframe, queue, macro_block, grain_size, assume_centered, hyperparameters_route
):
    from sklearnex.preview.covariance import EmpiricalCovariance

    X = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]])

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    empcov = EmpiricalCovariance(assume_centered=assume_centered)

    hyperparameters_route(queue, macro_block, grain_size)

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

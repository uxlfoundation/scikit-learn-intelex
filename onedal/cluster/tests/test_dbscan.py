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
from sklearn.cluster import DBSCAN as DBSCAN_SKLEARN
from sklearn.cluster.tests.common import generate_clustered_data

from onedal.cluster import DBSCAN as ONEDAL_DBSCAN
from onedal.tests.utils._device_selection import get_queues


def generate_data(
    low: int, high: int, samples_number: int, sample_dimension: tuple
) -> tuple:
    generator = np.random.RandomState()
    table_size = (samples_number, sample_dimension)
    return generator.uniform(low=low, high=high, size=table_size), generator.uniform(
        size=samples_number
    )


def check_labels_equals(left_labels: np.ndarray, right_labels: np.ndarray) -> bool:
    if left_labels.shape != right_labels.shape:
        raise Exception("Shapes not equal")
    if len(left_labels.shape) != 1:
        raise Exception("Shapes size not equals 1")
    if len(set(left_labels)) != len(set(right_labels)):
        raise Exception("Cluster counts not equal")
    dict_checker = {}
    for index_sample in range(left_labels.shape[0]):
        if left_labels[index_sample] not in dict_checker:
            dict_checker[left_labels[index_sample]] = right_labels[index_sample]
        elif dict_checker[left_labels[index_sample]] != right_labels[index_sample]:
            raise Exception("Wrong clustering")
    return True


def _test_dbscan_big_data_numpy_gen(
    queue,
    eps: float,
    min_samples: int,
    metric: str,
    use_weights: bool,
    low=-100.0,
    high=100.0,
    samples_number=1000,
    sample_dimension=4,
):
    data, weights = generate_data(
        low=low,
        high=high,
        samples_number=samples_number,
        sample_dimension=sample_dimension,
    )
    if use_weights is False:
        weights = None
    initialized_daal_dbscan = ONEDAL_DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric
    ).fit(X=data, sample_weight=weights, queue=queue)
    initialized_sklearn_dbscan = DBSCAN_SKLEARN(
        metric=metric, eps=eps, min_samples=min_samples
    ).fit(X=data, sample_weight=weights)
    check_labels_equals(
        initialized_daal_dbscan.labels_, initialized_sklearn_dbscan.labels_
    )


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
    ],
)
@pytest.mark.parametrize("use_weights", [True, False])
@pytest.mark.parametrize("queue", get_queues())
def test_dbscan_big_data_numpy_gen(queue, metric, use_weights: bool):
    eps = 35.0
    min_samples = 6
    _test_dbscan_big_data_numpy_gen(
        queue, eps=eps, min_samples=min_samples, metric=metric, use_weights=use_weights
    )


def _test_across_grid_parameter_numpy_gen(queue, metric, use_weights: bool):
    eps_begin = 0.05
    eps_end = 0.5
    eps_step = 0.05
    min_samples_begin = 5
    min_samples_end = 15
    min_samples_step = 1
    for eps in np.arange(eps_begin, eps_end, eps_step):
        for min_samples in range(min_samples_begin, min_samples_end, min_samples_step):
            _test_dbscan_big_data_numpy_gen(
                queue,
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                use_weights=use_weights,
            )


@pytest.mark.parametrize(
    "metric",
    [
        "euclidean",
    ],
)
@pytest.mark.parametrize("use_weights", [True, False])
@pytest.mark.parametrize("queue", get_queues())
def test_across_grid_parameter_numpy_gen(queue, metric, use_weights: bool):
    _test_across_grid_parameter_numpy_gen(queue, metric=metric, use_weights=use_weights)

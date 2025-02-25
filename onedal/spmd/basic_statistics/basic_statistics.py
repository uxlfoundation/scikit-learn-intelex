# ==============================================================================
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
# ==============================================================================

from ..._device_offload import support_input_format
from ...basic_statistics import BasicStatistics as BasicStatistics_Batch
from ...common._backend import bind_spmd_backend


class BasicStatistics(BasicStatistics_Batch):
    @bind_spmd_backend("basic_statistics")
    def compute(self, data, weights=None): ...

    @support_input_format
    def fit(self, data, sample_weight=None, queue=None):
        return super().fit(data, sample_weight, queue=queue)

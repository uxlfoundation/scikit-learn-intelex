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

from ...linear_model import (
    IncrementalLinearRegression as base_IncrementalLinearRegression,
)
from ..._device_offload import support_input_format
from .._base import BaseEstimatorSPMD


class IncrementalLinearRegression(BaseEstimatorSPMD, base_IncrementalLinearRegression):
    @support_input_format()
    def partial_fit(self, X, y, queue=None):
        return super().partial_fit(X, y, queue=queue)

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
from ...common._backend import bind_spmd_backend
from ...linear_model import LinearRegression as LinearRegression_Batch


class LinearRegression(LinearRegression_Batch):

    @bind_spmd_backend("linear_model.regression")
    def train(self, *args, **kwargs): ...

    @bind_spmd_backend("linear_model.regression")
    def finalize_train(self, *args, **kwargs): ...

    @bind_spmd_backend("linear_model.regression")
    def infer(self, params, model, X): ...

    @support_input_format
    def fit(self, X, y, queue=None):
        return super().fit(X, y, queue=queue)

    @support_input_format
    def predict(self, X, queue=None):
        return super().predict(X, queue=queue)

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
from ...decomposition.pca import PCA as PCABatch


class PCA(PCABatch):

    @bind_spmd_backend("decomposition.dim_reduction")
    def train(self, params, X, queue=None): ...

    @bind_spmd_backend("decomposition.dim_reduction")
    def finalize_train(self, *args, **kwargs): ...

    @support_input_format
    def fit(self, X, y=None, queue=None):
        return super().fit(X, queue=queue)

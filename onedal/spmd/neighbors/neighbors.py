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

from ..._device_offload import support_input_format, supports_queue
from ...common._backend import bind_spmd_backend
from ...neighbors import KNeighborsClassifier as KNeighborsClassifier_Batch
from ...neighbors import KNeighborsRegressor as KNeighborsRegressor_Batch


class KNeighborsClassifier(KNeighborsClassifier_Batch):

    @bind_spmd_backend("neighbors.classification")
    def train(self, *args, **kwargs): ...

    @bind_spmd_backend("neighbors.classification")
    def infer(self, *args, **kwargs): ...

    @support_input_format
    def fit(self, X, y, queue=None):
        return super().fit(X, y, queue=queue)

    @support_input_format
    def predict(self, X, queue=None):
        return super().predict(X, queue=queue)

    @support_input_format
    def predict_proba(self, X, queue=None):
        raise NotImplementedError("predict_proba not supported in distributed mode.")

    @support_input_format
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return super().kneighbors(X, n_neighbors, return_distance, queue=queue)


class KNeighborsRegressor(KNeighborsRegressor_Batch):

    @bind_spmd_backend("neighbors.search", lookup_name="train")
    def train_search(self, *args, **kwargs): ...

    @bind_spmd_backend("neighbors.search", lookup_name="infer")
    def infer_search(self, *args, **kwargs): ...

    @bind_spmd_backend("neighbors.regression")
    def train(self, *args, **kwargs): ...

    @bind_spmd_backend("neighbors.regression")
    def infer(self, *args, **kwargs): ...

    @support_input_format
    @supports_queue
    def fit(self, X, y, queue=None):
        if queue is not None and queue.sycl_device.is_gpu:
            return self._fit(X, y)
        else:
            raise ValueError(
                "SPMD version of kNN is not implemented for "
                "CPU. Consider running on it on GPU."
            )

    @support_input_format
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        return super().kneighbors(X, n_neighbors, return_distance, queue=queue)

    @support_input_format
    @supports_queue
    def predict(self, X, queue=None):
        return self._predict_gpu(X)

    def _get_onedal_params(self, X, y=None):
        params = super()._get_onedal_params(X, y)
        if "responses" not in params["result_option"]:
            params["result_option"] += "|responses"
        return params

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
from ...neighbors import NearestNeighbors as NearestNeighbors_Batch


class KNeighborsClassifier(KNeighborsClassifier_Batch):

    @bind_spmd_backend("neighbors.classification")
    def train(self, *args, **kwargs): ...

    @bind_spmd_backend("neighbors.classification")
    def infer(self, *args, **kwargs): ...

    @support_input_format
    def fit(self, X, y, queue=None):
        # Store queue to use during inference if not provided (if X is none in kneighbors)
        self.spmd_queue_ = queue
        return super().fit(X, y, queue=queue)

    @support_input_format
    @supports_queue
    def predict(self, X, queue=None):
        # SPMD classification: call the SPMD backend's inference directly
        from ...common._estimator_checks import _check_is_fitted
        from ...datatypes import from_table, to_table

        _check_is_fitted(self)

        # Use the SPMD backend for classification prediction
        params = super()._get_onedal_params(X)
        if "responses" not in params["result_option"]:
            params["result_option"] += "|responses"

        # Call SPMD classification inference backend
        X_table = to_table(
            X, queue=queue if queue else getattr(self, "spmd_queue_", None)
        )
        params["fptype"] = X_table.dtype
        result = self.infer(params, self._onedal_model, X_table)

        responses = from_table(result.responses, like=X)
        # Use slicing for dpctl compatibility (dpctl arrays don't have .ravel())
        if hasattr(responses, "ravel"):
            return responses.ravel()
        else:
            return responses[:, 0] if responses.ndim > 1 else responses

    @support_input_format
    def predict_proba(self, X, queue=None):
        raise NotImplementedError("predict_proba not supported in distributed mode.")

    @support_input_format
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        if X is None and queue is None:
            queue = getattr(self, "spmd_queue_", None)
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
        # Store queue to use during inference if not provided (if X is none in kneighbors)
        self.spmd_queue_ = queue
        if queue is not None and queue.sycl_device.is_gpu:
            return self._fit(X, y)
        else:
            raise ValueError(
                "SPMD version of kNN is not implemented for "
                "CPU. Consider running on it on GPU."
            )

    @support_input_format
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        if queue is None:
            queue = getattr(self, "spmd_queue_", None)
        return super().kneighbors(X, n_neighbors, return_distance, queue=queue)

    @support_input_format
    @supports_queue
    def predict(self, X, queue=None):
        # SPMD regression: call the SPMD backend's inference directly
        from ...common._estimator_checks import _check_is_fitted
        from ...datatypes import from_table, to_table

        _check_is_fitted(self)

        # Use the SPMD backend for regression prediction
        params = super()._get_onedal_params(X)
        if "responses" not in params["result_option"]:
            params["result_option"] += "|responses"

        # Call SPMD regression inference backend
        X_table = to_table(
            X, queue=queue if queue else getattr(self, "spmd_queue_", None)
        )
        params["fptype"] = X_table.dtype
        result = self.infer(params, self._onedal_model, X_table)

        responses = from_table(result.responses, like=X)
        # Use slicing for dpctl compatibility (dpctl arrays don't have .ravel())
        if hasattr(responses, "ravel"):
            return responses.ravel()
        else:
            return responses[:, 0] if responses.ndim > 1 else responses

    def _get_onedal_params(self, X, y=None):
        params = super()._get_onedal_params(X, y)
        if "responses" not in params["result_option"]:
            params["result_option"] += "|responses"
        return params


class NearestNeighbors(NearestNeighbors_Batch):

    @bind_spmd_backend("neighbors.search")
    def train(self, *args, **kwargs): ...

    @bind_spmd_backend("neighbors.search")
    def infer(self, *args, **kwargs): ...

    @support_input_format
    def fit(self, X, y=None, queue=None):
        # Store queue to use during inference if not provided (if X is none in kneighbors)
        self.spmd_queue_ = queue
        return super().fit(X, y, queue=queue)

    @support_input_format
    def kneighbors(self, X=None, n_neighbors=None, return_distance=True, queue=None):
        if X is None and queue is None:
            queue = getattr(self, "spmd_queue_", None)
        return super().kneighbors(X, n_neighbors, return_distance, queue=queue)

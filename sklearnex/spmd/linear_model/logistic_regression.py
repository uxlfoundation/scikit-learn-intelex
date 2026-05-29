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

from onedal.spmd.linear_model import LogisticRegression as onedal_LogisticRegression

from ...linear_model import LogisticRegression as LogisticRegression_Batch


class LogisticRegression(LogisticRegression_Batch):
    __doc__ = LogisticRegression_Batch.__doc__
    _onedal_LogisticRegression = staticmethod(onedal_LogisticRegression)

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        if queue is None or queue.sycl_device.is_cpu:
            # We don't use onedal backend for CPU, so we need an additional check here
            raise RuntimeError("Executing functions from SPMD backend requires a queue")
        return super()._onedal_fit(X, y, sample_weight=sample_weight, queue=queue)

    def _error_out_on_incompatible_devices(self, X, method_name: str) -> None:
        # custom function in LogisticRegression which will trigger for cpu data
        raise RuntimeError("Executing functions from SPMD backend requires a queue")
    
    def _onedal_score(self, X, y, sample_weight=None, queue=None):
        raise RuntimeError(
            "score method is not supported for LogisticRegression SPMD estimator."
        )

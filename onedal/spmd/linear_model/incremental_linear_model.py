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


from ...common._backend import DefaultPolicyOverride, bind_spmd_backend
from ...linear_model import (
    IncrementalLinearRegression as base_IncrementalLinearRegression,
)


class IncrementalLinearRegression(base_IncrementalLinearRegression):
    """
    Distributed incremental Linear Regression oneDAL implementation.

    API is the same as for `onedal.linear_model.IncrementalLinearRegression`.
    """

    @bind_spmd_backend("linear_model")
    def _get_policy(self): ...

    @bind_spmd_backend("linear_model.regression")
    def finalize_train(self, *args, **kwargs): ...

    def partial_fit(self, X, y, queue):
        # partial fit performed by parent backend, therefore default policy required
        with DefaultPolicyOverride(self):
            return super().partial_fit(X, y, queue)

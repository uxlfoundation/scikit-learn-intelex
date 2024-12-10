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

from daal4py.sklearn._utils import get_dtype

from ...basic_statistics import (
    IncrementalBasicStatistics as base_IncrementalBasicStatistics,
)
from ...common._backend import bind_default_backend, bind_spmd_backend
from ...datatypes import _convert_to_supported, to_table


class IncrementalBasicStatistics(base_IncrementalBasicStatistics):
    @bind_spmd_backend("basic_statistics")
    def compute(self, *args, **kwargs): ...

    @bind_spmd_backend("basic_statistics")
    def finalize_compute(self, *args, **kwargs): ...

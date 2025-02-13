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

from .base import (
    DTYPES,
    PATCHED_FUNCTIONS,
    PATCHED_MODELS,
    SPECIAL_INSTANCES,
    UNPATCHED_FUNCTIONS,
    UNPATCHED_MODELS,
    DummyEstimator,
    _get_processor_info,
    call_method,
    gen_dataset,
    gen_models_info,
    gen_sparse_dataset,
    sklearn_clone_dict,
)

__all__ = [
    "DTYPES",
    "PATCHED_FUNCTIONS",
    "PATCHED_MODELS",
    "UNPATCHED_FUNCTIONS",
    "UNPATCHED_MODELS",
    "SPECIAL_INSTANCES",
    "call_method",
    "gen_models_info",
    "gen_dataset",
    "gen_sparse_dataset",
    "sklearn_clone_dict",
    "DummyEstimator",
]

_IS_INTEL = "GenuineIntel" in _get_processor_info()

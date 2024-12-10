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

from onedal import _spmd_backend

if _spmd_backend is not None:

    class _SPMDDataParallelInteropPolicy(_spmd_backend.spmd_data_parallel_policy):
        def __init__(self, queue):
            self._queue = queue
            super().__init__(self._queue)

    def _get_spmd_policy(queue):
        # TODO:
        # cases when queue is None
        return _SPMDDataParallelInteropPolicy(queue)

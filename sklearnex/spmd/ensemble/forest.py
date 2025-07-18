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

from onedal.spmd.ensemble import RandomForestClassifier as onedal_RandomForestClassifier
from onedal.spmd.ensemble import RandomForestRegressor as onedal_RandomForestRegressor

from ...ensemble import RandomForestClassifier as RandomForestClassifier_Batch
from ...ensemble import RandomForestRegressor as RandomForestRegressor_Batch


class RandomForestClassifier(RandomForestClassifier_Batch):
    __doc__ = RandomForestClassifier_Batch.__doc__
    _onedal_factory = onedal_RandomForestClassifier

    def __init__(self, *args, local_trees_mode=False, **kwargs):
        self.local_trees_mode = local_trees_mode
        super().__init__(*args, local_trees_mode=local_trees_mode, **kwargs)

    def _onedal_cpu_supported(self, method_name, *data):
        # TODO:
        # check which methods supported SPMD interface on CPU.
        ready = super()._onedal_cpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready

    def _onedal_gpu_supported(self, method_name, *data):
        ready = super()._onedal_gpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready


class RandomForestRegressor(RandomForestRegressor_Batch):
    __doc__ = RandomForestRegressor_Batch.__doc__
    _onedal_factory = onedal_RandomForestRegressor

    def __init__(self, *args, local_trees_mode=False, **kwargs):
        self.local_trees_mode = local_trees_mode
        super().__init__(*args, local_trees_mode=local_trees_mode, **kwargs)

    def _onedal_cpu_supported(self, method_name, *data):
        # TODO:
        # check which methods supported SPMD interface on CPU.
        ready = super()._onedal_cpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready

    def _onedal_gpu_supported(self, method_name, *data):
        ready = super()._onedal_gpu_supported(method_name, *data)
        if not ready:
            raise RuntimeError(
                f"Method {method_name} in {self.__class__.__name__} "
                "is not supported with given inputs."
            )
        return ready

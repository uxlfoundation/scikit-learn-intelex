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

from onedal.ensemble import RandomForestClassifier as RandomForestClassifier_Batch
from onedal.ensemble import RandomForestRegressor as RandomForestRegressor_Batch

from ...common._backend import bind_spmd_backend


class RandomForestClassifier(RandomForestClassifier_Batch):
    @bind_spmd_backend("decision_forest.classification")
    def train(self, *args, **kwargs): ...

    @bind_spmd_backend("decision_forest.classification")
    def infer(self, *args, **kwargs): ...


class RandomForestRegressor(RandomForestRegressor_Batch):
    @bind_spmd_backend("decision_forest.regression")
    def train(self, *args, **kwargs): ...

    @bind_spmd_backend("decision_forest.regression")
    def infer(self, *args, **kwargs): ...

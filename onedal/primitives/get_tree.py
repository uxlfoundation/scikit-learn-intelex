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

from onedal import _default_backend, _dpc_backend

backend = _dpc_backend or _default_backend


def get_tree_state_cls(model, iTree, n_classes):
    return backend.get_tree.classification.get_tree_state(model, iTree, n_classes)


def get_tree_state_reg(model, iTree, n_classes):
    return backend.get_tree.regression.get_tree_state(model, iTree, n_classes)

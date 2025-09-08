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

from onedal.spmd.covariance import EmpiricalCovariance as onedal_EmpiricalCovariance

from ...preview.covariance import EmpiricalCovariance as EmpiricalCovariance_Batch


class EmpiricalCovariance(EmpiricalCovariance_Batch):
    __doc__ = EmpiricalCovariance_Batch.__doc__
    _onedal_covariance = staticmethod(onedal_EmpiricalCovariance)

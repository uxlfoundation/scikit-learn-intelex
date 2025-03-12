# ===============================================================================
# Copyright 2021 Intel Corporation
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
# ===============================================================================

from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel as _sklearn_rbf_kernel

from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.metrics import pairwise_distances
from onedal._device_offload import support_input_format
from onedal.primitives import rbf_kernel as _onedal_rbf_kernel

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain

pairwise_distances = support_input_format(freefunc=True, queue_param=False)(
    pairwise_distances
)


if sklearn_check_version("1.6"):
    from sklearn.utils.validation import validate_data
else:
    validate_data = BaseEstimator._validate_data


class RBFKernel:
    __doc__ = _sklearn_rbf_kernel.__doc__

    def __init__(self):
        pass

    def _onedal_supported(self, method_name, *data):
        patching_status = PatchingConditionsChain(
            f"sklearn.metrics.pairwise.{method_name}"
        )
        return patching_status

    def _onedal_cpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_gpu_supported(self, method_name, *data):
        return self._onedal_supported(method_name, *data)

    def _onedal_rbf_kernel(self, X, Y=None, gamma=None, queue=None):
        return _onedal_rbf_kernel(X, Y, gamma, queue)

    def compute(self, X, Y=None, gamma=None):
        result = dispatch(
            self,
            "rbf_kernel",
            {
                "onedal": self.__class__._onedal_rbf_kernel,
                "sklearn": _sklearn_rbf_kernel,
            },
            X,
            Y,
            gamma,
        )

        return result


def rbf_kernel(X, Y=None, gamma=None):
    return RBFKernel().compute(X, Y, gamma)

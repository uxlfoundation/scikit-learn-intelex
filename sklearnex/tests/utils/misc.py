# ==============================================================================
# Copyright contributors to the oneDAL project
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
from daal4py.sklearn._utils import sklearn_check_version

if sklearn_check_version("1.9"):
    from sklearn.utils._array_api import get_namespace_and_device

    __all__ = ["assert_same_namespace"]

    def assert_same_namespace(a, b):
        xp_a, _, device_a = get_namespace_and_device(a)
        xp_b, _, device_b = get_namespace_and_device(b)
        assert xp_a == xp_b
        assert device_a == device_b

else:
    __all__ = []

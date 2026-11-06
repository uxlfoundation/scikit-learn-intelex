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
    import warnings

    from sklearn.utils._array_api import get_namespace_and_device

    __all__ = [
        "assert_same_namespace",
        "assert_public_attributes_are_in_namespace",
        "assert_same_attributes",
    ]

    def assert_same_namespace(a, b):
        xp_a, _, device_a = get_namespace_and_device(a)
        xp_b, _, device_b = get_namespace_and_device(b)
        assert xp_a == xp_b
        assert device_a == device_b

    def check_is_array(obj) -> bool:
        return hasattr(obj, "__dlpack__") or hasattr(obj, "shape")

    # TODO: revisit these attributes once scikit-learn 1.10 decides
    # on the logic for their type and what happens after moving estimators.
    UNMOVABLE_ATTRIBUTES = [
        "classes_",
        "feature_names_in_",
    ]

    def assert_public_attributes_are_in_namespace(estimator, xp, device):
        for attr in dir(estimator):
            if (
                attr not in UNMOVABLE_ATTRIBUTES
                and not attr.startswith("_")
                and attr.endswith("_")
                and check_is_array(getattr(estimator, attr))
            ):
                xp_obj, _, device_obj = get_namespace_and_device(getattr(estimator, attr))
                assert xp_obj == xp
                assert device_obj == device

    def assert_same_attributes(obj_check, obj_reference):
        # Warning is thrown whenever oneDAL hyperparameters (like block sizes)
        # are accessed through Python
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            attrs_check = dir(obj_check)
            attrs_reference = dir(obj_reference)
        for attr in attrs_reference:
            assert (
                attr in attrs_check
            ), f"Attribute '{attr}' from reference missing in object to check"

else:
    __all__ = []

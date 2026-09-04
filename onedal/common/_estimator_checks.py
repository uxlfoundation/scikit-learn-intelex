# ===============================================================================
# Copyright 2022 Intel Corporation
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


def _check_is_fitted(estimator, attributes=None, *, msg=None):
    if msg is None:
        msg = (
            "This %(name)s instance is not fitted yet. Call 'fit' with "
            "appropriate arguments before using this estimator."
        )

    if not (
        hasattr(estimator, "fit")
        or (hasattr(estimator, "partial_fit") and hasattr(estimator, "finalize_fit"))
    ):
        raise TypeError("%s is not an estimator instance." % (estimator))

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = all([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise AttributeError(msg % {"name": type(estimator).__name__})

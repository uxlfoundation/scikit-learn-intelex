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

import daal4py.sklearn


def test_monkey_patching():
    _tokens = daal4py.sklearn.sklearn_patch_names()
    _values = daal4py.sklearn.sklearn_patch_map().values()
    _classes = list()
    for v in _values:
        _classes.append(v[0])

    assert len(_tokens) == len(_classes)
    assert (
        isinstance(_tokens, list) and len(_tokens) > 0
    ), "Internal Error: list of patched names has unexcepable format."

    daal4py.sklearn.patch_sklearn()

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        class_module = getattr(p, n).__module__
        assert class_module.startswith("daal4py"), "Patching has completed with error."

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        daal4py.sklearn.unpatch_sklearn(t)
        class_module = getattr(p, n).__module__
        assert class_module.startswith("sklearn"), "Unpatching has completed with error."

    daal4py.sklearn.unpatch_sklearn()

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        class_module = getattr(p, n).__module__
        assert class_module.startswith("sklearn"), "Unpatching has completed with error."

    for i, _ in enumerate(_tokens):
        t = _tokens[i]
        p = _classes[i][0]
        n = _classes[i][1]

        daal4py.sklearn.patch_sklearn(t)

        class_module = getattr(p, n).__module__
        assert class_module.startswith("daal4py"), "Patching has completed with error."

    daal4py.sklearn.unpatch_sklearn()

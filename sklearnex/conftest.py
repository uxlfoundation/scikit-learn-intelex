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

import io
import logging

import pytest

from onedal.tests.utils._dataframes_support import array_api_frameworks
from sklearnex import config_context, patch_sklearn, unpatch_sklearn


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "allow_sklearn_fallback: mark test to not check for sklearnex usage"
    )
    config.addinivalue_line(
        "markers", "mpi: mark test to require MPI for distributed testing"
    )


@pytest.fixture(autouse=True)
def allow_sklearn_fallback(request):
    with config_context(
        allow_sklearn_fallback=request.node.get_closest_marker("allow_sklearn_fallback")
    ):
        yield


@pytest.fixture(autouse=True)
def _array_api_dispatch_for_device_frameworks(request):
    # Without array_api_dispatch, ``_device_offload.dispatch`` host-transfers these
    # inputs and ``support_input_format``/``wrap_output_data`` restore the namespace
    # via ``__array_namespace__``, which torch lacks -- so the on-device paths go
    # untested and torch results come back as numpy. Force dispatch to exercise the
    # real array API path; allow_sklearn_fallback tests cover host paths on purpose.
    dataframe = getattr(request.node, "callspec", None)
    dataframe = dataframe.params.get("dataframe") if dataframe else None
    if dataframe in array_api_frameworks and not request.node.get_closest_marker(
        "allow_sklearn_fallback"
    ):
        with config_context(array_api_dispatch=True):
            yield
    else:
        yield


@pytest.fixture
def with_sklearnex():
    patch_sklearn()
    yield
    unpatch_sklearn()


@pytest.fixture
def with_array_api():
    with config_context(array_api_dispatch=True):
        yield


@pytest.fixture
def without_allow_sklearn_after_onedal():
    with config_context(allow_sklearn_after_onedal=False):
        yield

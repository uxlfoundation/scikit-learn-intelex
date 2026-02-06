#! /usr/bin/env python
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

import os
import subprocess
import sys

import pytest

# This is a workaround for older versions of Python on Windows
# which didn't have it as part of the built-in 'os' module.
EX_OK = os.EX_OK if hasattr(os, "EX_OK") else 0

# Note: from the structure of this file, one might think of adding a test

# along the lines of 'test_patching_all_from_command_line'. There is however
# an issue in that, after the first time a scikit-learn module is imported,
# further calls to 'patch_sklearn' with different arguments will have no effect
# since sklearn is already imported. Reloading it through 'importlib.reload'
# or deleting it from 'sys.modules' doesn't appear to have the intended effect
# either. This also makes this first command-line fixture and tests that use
# it not entirely idempotent, given that they need to import sklearn modules.

# Note 2: don't try to change these into 'yield' fixtures, because otherwise
# some test runners on windows which use multi-processing will throw errors
# about the fixtures not being serializable.


@pytest.fixture
def patch_svc_from_command_line(request):
    err_code = subprocess.call(
        [sys.executable, "-m", "sklearnex.glob", "patch_sklearn", "-a", "sklearn.svm.SVC"]
    )
    assert err_code == EX_OK

    def finalizer():
        err_code = subprocess.call(
            [
                sys.executable,
                "-m",
                "sklearnex.glob",
                "unpatch_sklearn",
                "-a",
                "sklearn.svm.SVC",
            ]
        )
        assert err_code == EX_OK

    request.addfinalizer(finalizer)
    return


def test_patching_svc_from_command_line(patch_svc_from_command_line):
    from sklearn.svm import SVC, SVR

    assert SVC.__module__.startswith("sklearnex")
    assert not SVR.__module__.startswith("daal4py") and not SVR.__module__.startswith(
        "sklearnex"
    )


def test_unpatching_svc_from_command_line(patch_svc_from_command_line):
    err_code = subprocess.call(
        [sys.executable, "-m", "sklearnex.glob", "unpatch_sklearn"]
    )
    assert err_code == EX_OK
    from sklearnex import unpatch_sklearn

    unpatch_sklearn()
    from sklearn.svm import SVC, SVR

    assert not SVC.__module__.startswith("daal4py") and not SVC.__module__.startswith(
        "sklearnex"
    )
    assert not SVR.__module__.startswith("daal4py") and not SVR.__module__.startswith(
        "sklearnex"
    )


@pytest.fixture
def patch_svc_from_function(request):
    from sklearnex import patch_sklearn, unpatch_sklearn

    patch_sklearn(name=["sklearn.svm.SVC"], global_patch=True)

    def finalizer():
        unpatch_sklearn(name=["sklearn.svm.SVC"], global_unpatch=True)

    request.addfinalizer(finalizer)
    return


def test_patching_svc_from_function(patch_svc_from_function):
    from sklearn.svm import SVC, SVR

    assert SVC.__module__.startswith("sklearnex")
    assert not SVR.__module__.startswith("daal4py") and not SVR.__module__.startswith(
        "sklearnex"
    )


def test_unpatching_svc_from_function(patch_svc_from_function):
    from sklearnex import unpatch_sklearn

    unpatch_sklearn(global_unpatch=True)
    from sklearn.svm import SVC, SVR

    assert not SVC.__module__.startswith("daal4py")
    assert not SVC.__module__.startswith("sklearnex")
    assert not SVR.__module__.startswith("daal4py")
    assert not SVR.__module__.startswith("sklearnex")


@pytest.fixture
def patch_all_from_function(request):
    from sklearnex import patch_sklearn, unpatch_sklearn

    patch_sklearn(global_patch=True)

    def finalizer():
        unpatch_sklearn(global_unpatch=True)

    request.addfinalizer(finalizer)
    return


def test_patching_svc_from_function(patch_all_from_function):
    from sklearn.svm import SVC, SVR

    assert SVC.__module__.startswith("sklearnex")
    assert SVR.__module__.startswith("sklearnex")


def test_unpatching_all_from_function(patch_all_from_function):
    from sklearnex import unpatch_sklearn

    unpatch_sklearn(global_unpatch=True)
    from sklearn.svm import SVC, SVR

    assert not SVC.__module__.startswith("daal4py")
    assert not SVC.__module__.startswith("sklearnex")
    assert not SVR.__module__.startswith("daal4py")
    assert not SVR.__module__.startswith("sklearnex")

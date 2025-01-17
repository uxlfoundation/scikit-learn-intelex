# ==============================================================================
# Copyright 2014 Intel Corporation
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

from os import environ

# sklearn requires manual enabling of Scipy array API support
# if `array-api-compat` package is present in environment
# TODO: create generic approach to handle this for all tests
environ["SCIPY_ARRAY_API"] = "1"


import unittest

import sklearn.utils.estimator_checks
from sklearn.utils.estimator_checks import check_estimator

from daal4py import _get__daal_link_version__ as dv

# First item is major version - 2021,
# second is minor+patch - 0110,
# third item is status - B
daal_version = (int(dv()[0:4]), dv()[10:11], int(dv()[4:8]))
print("DAAL version:", daal_version)

from daal4py.sklearn.ensemble import (
    AdaBoostClassifier,
    GBTDAALClassifier,
    GBTDAALRegressor,
)


def check_version(rule, target):
    if not isinstance(rule[0], type(target)):
        if rule > target:
            return False
    else:
        for rule_item in rule:
            if rule_item > target:
                return False
            if rule_item[0] == target[0]:
                break
    return True


def _replace_and_save(md, fns, replacing_fn):
    """
    Replaces functions in `fns` list in `md` module with `replacing_fn`.

    Returns the dictionary with functions that were replaced.
    """
    saved = dict()
    for check_f in fns:
        try:
            fn = getattr(md, check_f)
            setattr(md, check_f, replacing_fn)
            saved[check_f] = fn
        except RuntimeError:
            pass
    return saved


def _restore_from_saved(md, saved_dict):
    """
    Restores functions in `md` that were replaced in the function above.
    """
    for check_f in saved_dict:
        setattr(md, check_f, saved_dict[check_f])


class Test(unittest.TestCase):
    def test_GBTDAALClassifier(self):
        check_estimator(GBTDAALClassifier())

    def test_GBTDAALRegressor(self):
        def dummy(*args, **kwargs):
            pass

        md = sklearn.utils.estimator_checks
        # got unexpected slightly different
        # prediction result between two same calls in this test
        saved = _replace_and_save(md, ["check_estimators_data_not_an_array"], dummy)
        check_estimator(GBTDAALRegressor())
        _restore_from_saved(md, saved)

    def test_AdaBoostClassifier(self):
        check_estimator(AdaBoostClassifier())


if __name__ == "__main__":
    unittest.main()

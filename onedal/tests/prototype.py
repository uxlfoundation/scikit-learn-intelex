# ==============================================================================
# Copyright Contributors to the oneDAL Project
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

""" This file describes necessary characteristics and design patterns of
onedal estimators.

This can be used as a foundation for developing other estimators. Most
comments guiding code development should be removed unless pertinent to the
implementation.  """

from .._device_offload import supports_queue
from ..common._backend import bind_default_backend
from ..datatypes import from_table, to_table


class PrototypeEstimator:

    def __init__(self, allow_nan=False):
        # The __init__ method should only assign class attributes matching
        # the input parameters (similar to sklearn).  It is not to assign
        # any attributes which aren't related to the operation of oneDAL.
        # This is means that it should not conform to sklearn, only to
        # oneDAL. Don't add unnecessary attributes which only match sklearn,
        # these should be translated by the sklearnex estimator. In this case
        # the only parameter for the finiteness checker is the 'allow_nan'
        # param.
        self.allow_nan = allow_nan

    # see documentation on bind_default_backend. There exists three possible
    # oneDAL pybind11 interfaces, 'host', 'dpc' and 'spmd'. These are for
    # cpu-only, cpu and gpu, and multi-device computation respectively. Logic
    # in the onedal module will determine which can be used at import time.
    # It will attempt to use the `dpc` interface if possible (which enables
    # gpu computation) but requires a SYCL runtime. If not possible it will
    # silently fall back to the 'host' pybind11 interface. The backend
    # binding logic will seamlessly handle this for the estimator. The 'spmd'
    # backend is specific to onedal estimators defined in the 'spmd' folder.
    # The binding makes the pybind11 function a method of this class with
    # the same name (in this case ProtoTypeEstimator.compute should call
    # the pybind11 function onedal.backend.finiteness_checker.compute.compute)
    # where backend can be one of 'host', 'dpc' or 'spmd'.
    @bind_default_backend("finiteness_checker.compute")
    def compute(self, params, data_table): ...

    @supports_queue
    def fit(self, X, y, queue=None):
        # convert the data to oneDAL tables in preparation for use by the
        # oneDAL pybind11 interfaces/objects.
        X_t, y_t = to_table(X, y)

        # Generating the params dict can be centralized into a class method,
        # but it must be named ``_get_onedal_params``. Parameter 'fptype' is
        # specific to the pybind11 interface, and cannot be found in oneDAL
        # documentation. This tells oneDAL what float type to use for the
        # computation. The safest and best way to assign this value is after
        # the input data has been converted to a oneDAL table, as the dtype
        # is standardized (taken care of by ``to_table``).  This dtype is a
        # ``numpy`` dtype due to its ubiquity and native support in pybind11.
        params = {
            "fptype": X_t.dtype,
            "method": "dense",
            "allow_nan": self.allow_nan,
        }

        # This is the call to the oneDAL pybind11 backend, which was
        # previously bound using ``bind_default_backend``. It returns a
        # pybind11 Python interface to the oneDAL C++ result object.
        result_X = self.compute(params, X_t)
        # The use of ``compute`` here is unusual, as typical naming conventions
        # for oneDAL and sklearn would have ``fit`` match oneDALs ``train``,
        # and ``predict`` match oneDAL's ``infer``. Please refer to the oneDAL
        # design documentation to determine the best translation. Generally the
        # sklearn naming scheme for class methods should be used here, but
        # calls to the pybind11 interfaces should follow oneDAL naming.
        result_y = self.compute(params, y_t)

        # Oftentimes oneDAL table objects are attributes of the oneDAL C++
        # object. These can be converted into various common data frameworks
        # like ``numpy`` or ``dpctl.tensor`` using ``from_table``. In this
        # case the output is a basic python type (bool) which can be handled
        # easily just with pybind11 without any special code.
        # Attributes of the result object are copied to attributes of the
        # onedal estimator object.
        self.finite_ = result_X.finite and result_y.finite

        # These attributes are set in order to show the process of setting 
        # and returning array values and shouldn't normally be done.  In
        # setting return attributes, post processing of the values to match
        # needed for sklearn must occur in the sklearnex estimator.
        self.fit_X_ = from_table(X_t)
        self.fit_y_ = from_table(y_t)


    @supports_queue
    def predict(self, X, queue=None):
        X_t = to_table(X)
        params = {
            "fptype": X_t.dtype,
            "method": "dense",
            "allow_nan": self.allow_nan,
        }
        return self.compute(params, X_t).finite

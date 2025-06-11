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
sklearnex estimators.  

This can be used as a foundation for developing other estimators. Most
comments guiding code development should be used unless pertinent to the
implementation.  """
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import get_namespace
from ..utils.validation import validate_data

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import StrOptions


################
# IMPORT NOTES #
################
#
# 1) All sklearnex estimators must inherit oneDALestimator and the sklearn
# estimator that it is replicating (i.e. before in the mro).  If there is
# not an equivalent sklearn estimator, then sklearn's BaseEstimator must be
# inherited.
#
# 2) ``check_is_fitted`` is required for any method in an estimator which
# requires first calling ``fit`` or ``partial_fit``. This is a sklearn
# requirement.
#
# 2) Every estimator should be decorated by ``control_n_jobs`` to properly
# create parallelization control for the oneDAL library via the ``n_jobs``
# parameter. This parameter is added to ``__init__`` automatically.
#
# 3) For compatability reasons, ``daal_check_version`` and
# ``sklearn_check_version`` add or remove capabilites based on the installed
# oneDAL library and scikit-learn package. This is often necessary for the
# state of oneDAL development and scikit-learn characteristics. This should
# be used at import time instead of run time whenever possible/ practical.
#
# 4) ``dispatch`` is a key central function for evaluating data with either
# oneDAL or sklearn. All oneDAL algorithms which are to be directly used
# should be accessed via this function.
#
# 5) ``PatchingConditionsChain`` is used in conjunction with ``dispatch``
# and methods ``_onedal_cpu_supported`` and ``_onedal_gpu_supported`` to
# evaluate if the required evaluation on data is possible with oneDAL or
# sklearn.
#
# 6) ``get_namespace`` is key for array_api support, which yields the
# namespace associated with the given array for use in data conversion
# necessary for to and from oneDAL. An internal version is preferred due to
# limitations in sklearn versions and specific IntelPython data framework
# support (see dpctl tensors and dpnp).
#
# 7) ``validate_data`` checks data quality and estimator status before
# evaluating the function. This replicates a sklearn functionality with key
# performance changes implemented in oneDAL and therefore should only be
# imported from sklearnex and not sklearn.
#
# 8) All estimators require validation of the parameters given at
# initialization. This aspect was introduced in sklearn 1.2, any additional
# parameters must extend the dictionary for checking.

##########################
# METHOD HEIRARCHY NOTES #
##########################
#
# Sklearnex estimator methods can be thought of in 3 major tiers.
#
# Tier 1: Those methods which offload to oneDAL using ``dispatch``. Typical
# examples are ``fit`` and ``predict``. They use a direct equivalent oneDAL
# function for evaluation. These methods are of highest priority and have
# performance benchmark requirements.
#
# Tier 2: Those methods that use a Tier 1 method with additional Python
# calculations (usually a sklearn method or applied math function). Examples
# are ``score`` and ``predict_log_proba``. Oftentimes the additional
# calculations are trivial, meaning benchmarking is not required.
#
# Tier 3: Those methods which directly use sklearn functionality. Typically
# these can be directly inherited, but can be problematic with respect
# to other framework support. These can be wrapped with the sklearnex
# function ``wrap_output_data`` to guarantee array API, dpctl tensor, and
# dpnp support but should be addressed with care/guidance in a case-by-case
# basis.

########################
# CONTROL_N_JOBS NOTES #
########################
#
# All tier 1 methods should be in the decorated_methods list for oneDAL
# parallelism control. In general, changes to oneDAL parallelism should only
# be done once per public method call.This may mean some tier 2 methods
# must be added to the list along with some restructuring of the related 
# tier 1 methods. An illustrative example could be an estimator which
# implements ``fit_transform`` where combining ``fit`` and ``transform``
# tier 1 methods may set n_jobs twice.

@control_n_jobs(decorated_methods=["fit", "predict"])
class PrototypeEstimator(oneDALEstimator, BaseEstimator):
    # PrototypeEstimator is a sklearnex-only estimator, shown by the 
    # inheritance of BaseEstimator. oneDALEstimator for these estimators
    # without a sklearn equivalent must have oneDALEstimator directly
    # before in the mro.

########################
# GENERAL DESIGN NOTES #
########################
#
# Sklearnex estimators follow a Matryoshka doll pattern with respect to 
# the underlying oneDAL. The sklearnex estimator is a public-facing API
# which mimics sklearn. Sklearnex estimators will use another estimator,
# defined in the ``onedal`` module, to create a onedal estimator written in
# python. Finally, this python object will use pybind11 to call oneDAL
# directly via pybind11-generated objects and functions. These are separate
# entities and do not inherit from one another. The clear separation has
# utility so long that the following rules are followed:
#
# 1) All conformance to sklearn should happen in sklearnex estimators, with
# all variations between the supported sklearn versions handled there. This
# includes transforming result data into a format which matches sklearn.
# This is done to minimize and focus maintenance with respect to sklearn to
# the sklearnex module.
# 
# 2) The onedal estimator handles necessary data conversion and preparation
# for invoking calls to onedal. These objects should not be influenced by 
# sklearn design or have any sklearn version dependent characteristics. 
# Users should be able to use these objects directly to fit data without
# sklearn, giving the ability to use raw data directly and avoiding
# sklearn pre-processing checks as necessary.
#
# 3) Pybind11 interfaces should be not be made public to the user unless
# absolutely necessary, as operation there assumes checks in the other
# objects have been sufficiently carried out. In most circumstances, the
# pybind11 interface should be invoked by the python onedal estimator object.
#


    def __init__(self):
        # Object instantiation is strictly limited by sklearn. It is only
        # allowed to take the keyword arguments and store them as 
        # attributes with the same name. When replicating a sklearn
        # estimator, it may be possible to use the inherited version of
        # ``__init__`` from sklearn. The prototype uses defined parameters
        # to highlight the process in detail when adding new parameters to
        # an estimator.
        pass

############################
# TIER 1 METHOD FLOW NOTES #
############################
#
# Some knowledge of the process flow from the sklearnex perspective is
# necessary to understand how to implement an estimator. For Tier 1 methods,
# the general process is as follows:
#
# 1) If a method which requires a fitted estimator, the method must call
# ``check_is_fitted`` before calling ``dispatch``. This verifies that
# aspects of the fit are available for analysis (whether oneDAL may be used
# or not). 
#
# 2) ``dispatch`` is called. This takes the estimator object, method name,
# and the two possible evaluation branches and proceeds to call 
# ``_onedal_gpu_supported`` if a SYCL queue is found or set via the
# ``target_offload`` config. Otherwise ``_onedal_cpu_supported`` is called.
#
# 3) ``_onedal_gpu_supported`` or ``_onedal_cpu_supported`` creates a
# PatchingConditionsChain object, takes the input data and estimator
# parameters, and evaluates whether the estimator and data can be run using
# oneDAL. This information is logged to the `sklearnex` logger.
#
# 4) Either sklearn is called, or a object from onedal is created and called
# using the input data. This process is handled in a function which has the
# prefix "_onedal_" followed by the method name. When fitting data, the 
# returned onedal estimator object is stored as the ``_onedal_estimator``
# attribute.
#
# 5) If necessary result data is returned from the estimator. Attributes
# from the onedal estimator are copied over to the sklearnex estimator.
# Result data is returned to user.

    def fit(self, X, y):
        # 
        pass

    def predict(self, X):
        pass

    def _onedal_fit(self, X, y):

    def _onedal_predict(self, X, y):

    def _onedal_cpu_supported(self, method):
        pass

    def _onedal_gpu_supported(self, method):
        pass

    def score(self, X, y):
        pass

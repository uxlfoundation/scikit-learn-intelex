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

"""Sklearnex module estimator design guide and example.

This can be used as a foundation for developing other estimators. Most
comments guiding code development should be removed if reused unless
pertinent to the derivative implementation."""
import numpy as np
import scipy.sparse as sp
from sklearn.dummy import DummyRegressor as _sklearn_DummyRegressor
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal._device_offload import support_input_format
from onedal.tests.prototype import DummyEstimator as onedal_DummyEstimator

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import validate_data

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
# 3) Every estimator should be decorated by ``control_n_jobs`` to properly
# create parallelization control for the oneDAL library via the ``n_jobs``
# parameter. This parameter is added to ``__init__`` automatically.
#
# 4) For compatibility reasons, ``daal_check_version`` and
# ``sklearn_check_version`` add or remove capabilities based on the installed
# oneDAL library and scikit-learn package. This is often necessary for the
# state of oneDAL development and scikit-learn characteristics. This should
# be used at import time instead of run time whenever possible/ practical.
#
# 5) If a sklearn estimator is imported, it must have the ``_sklearn_``
# prefix added upon import in order to prevent its discovery, highlight
# its nature as private, and prevent a namespace collision. Any onedal
# imported estimator should similarly  have the ``onedal_`` prefix added
# (as it should have the same name as the sklearnex estimator).
#
# 6) ``dispatch`` is a key central function for evaluating data with either
# oneDAL or sklearn. All oneDAL algorithms which are to be directly used
# should be accessed via this function. It should not be used unless a
# call to a onedal estimator occurs.
#
# 7) ``PatchingConditionsChain`` is used in conjunction with ``dispatch``
# and methods ``_onedal_cpu_supported`` and ``_onedal_gpu_supported`` to
# evaluate if the required evaluation on data is possible with oneDAL or
# sklearn.
#
# 8) ``get_namespace`` is key for array_api support, which yields the
# namespace associated with the given array for use in data conversion
# necessary for to and from oneDAL. An internal version is preferred due to
# limitations in sklearn versions and specific DPEX data framework support
# (see dpctl tensors and dpnp).
#
# 9) ``validate_data`` checks data quality and estimator status before
# evaluating the function. This replicates a sklearn functionality with key
# performance changes implemented in oneDAL and therefore should only be
# imported from sklearnex and not sklearn.
#
# 10) All estimators require validation of the parameters given at
# initialization. This aspect was introduced in sklearn 1.2, any additional
# parameters must extend the dictionary for checking.  This validation
# normally occurs in the ``fit`` method.
#

##########################
# METHOD HIERARCHY NOTES #
##########################
#
# Sklearnex estimator methods can be thought of in 3 major tiers.
#
# Tier 1: Methods which offload to oneDAL using ``dispatch``. Typical
# examples are ``fit`` and ``predict``. They use a direct equivalent oneDAL
# function for evaluation. These methods are of highest priority and have
# performance benchmark requirements.
#
# Tier 2: Methods that use a Tier 1 method with additional Python
# calculations (usually a sklearn method or applied math function). Examples
# are ``score`` and ``predict_log_proba``. Oftentimes the additional
# calculations are trivial, meaning benchmarking is not required.
#
# Tier 3: Methods which directly use sklearn functionality. Typically these
# can be directly inherited, but can be problematic with respect to other
# framework support. These can be wrapped with the sklearnex function
# ``wrap_output_data`` to guarantee array API, dpctl tensor, and dpnp
# support but should be addressed with care/guidance in a case-by-case
# basis.
#
# When the sklearnex method is replacing an inherited sklearn method, it
# must match the method signature exactly. For sklearnex-only estimators,
# attempt to match convention to sklearn estimators which are closely related.

########################
# CONTROL_N_JOBS NOTES #
########################
#
# All tier 1 methods should be in the decorated_methods list for oneDAL
# parallelism control. In general, changes to oneDAL parallelism should only
# be done once per public method call. This may mean some tier 2 methods
# must be added to the list along with some restructuring of the related
# tier 1 methods. An illustrative example could be an estimator which
# implements ``fit_transform`` where combining ``fit`` and ``transform``
# tier 1 methods may set n_jobs twice.


# enable_array_api enables the sklearnex code to work with and directly pass
# array_api and dpep frameworks data (dpnp, dpctl tensors, and pytorch for
# example) to the oneDAL backend
@enable_array_api
@control_n_jobs(decorated_methods=["fit", "predict"])
class DummyRegressor(oneDALEstimator, _sklearn_DummyRegressor):
    # All sklearnex estimators must inherit a sklearn estimator, sklearnex-
    # only estimators are shown by the inheritance of sklearn's
    # BaseEstimator. Additionally, inherited oneDALEstimator for estimators
    # without a sklearn equivalent must occur directly before BaseEstimator
    # in the mro.

    ##################################
    # GENERAL ESTIMATOR DESIGN NOTES #
    ##################################
    #
    # As a rule conform to sklearn design rules as much as possible
    # (https://scikit-learn.org/stable/developers/develop.html)
    # This includes inheriting the proper sklearn Mixin classes depending
    # on the sklearnex estimator functionality.
    #
    # All estimators should be defined in a Python file located in a folder
    # limited to the folder names in this directory:
    # https://github.com/scikit-learn/scikit-learn/tree/main/sklearn
    # All estimators should be properly added into the patching map located
    # in sklearnex/dispatcher.py following the convention made there. This
    # is important for having the estimator properly tested and available
    # in sklearn.
    #
    # Sklearnex estimators follow a Matryoshka doll pattern with respect to
    # the underlying oneDAL library. The sklearnex estimator is a
    # public-facing API which mimics sklearn. Sklearnex estimators will
    # create another estimator, defined in the ``onedal`` module, for
    # having a Python interface with oneDAL. Finally, this Python object
    # will use pybind11 to call oneDAL directly via pybind11-generated
    # objects and functions This is known as the ``backend``. These are
    # separate entities and do not inherit from one another. The clear
    # separation has utility so long that the following rules are followed:
    #
    # 1) All conformance to sklearn should happen in sklearnex estimators,
    # with all variations between the supported sklearn versions handled
    # there. This includes transforming result data into a format which
    # matches sklearn. This is done to minimize and focus maintenance with
    # respect to sklearn to the sklearnex module.
    #
    # 2) The onedal estimator handles necessary data conversion and
    # preparation for invoking calls to onedal. These objects should not be
    # influenced by sklearn design or have any sklearn version dependent
    # characteristics. Users should be able to use these objects directly
    # to fit data without sklearn, giving the ability to use raw data
    # directly and avoiding sklearn pre-processing checks as necessary.
    #
    # 3) Pybind11 interfaces should not be made public to the user unless
    # absolutely necessary, as operation there assumes checks in the other
    # objects have been sufficiently carried out. In most circumstances, the
    # pybind11 interface should be invoked by the Python onedal estimator
    # object.
    #
    # 4) If the estimator replicates/inherits from a sklearn estimator,
    # then only implemented public methods should be those which override
    # those from the sklearn estimator. The sklearn method should only be
    # overridden if an equivalent oneDAL-accelerated capability exists
    # following the tier system described below. If it is sklearnex only,
    # then it should try to follow sklearn conventions of sklearn estimators
    # which are most closely related (e.g. IncrementalPCA for incremental
    # estimators). NOTE: as per the sklearn design rules, all estimator
    # attributes with trailing underscores are return values and are of
    # some type of data (and therefore not themselves oneDAL-accelerated).
    #
    # Information about the onedal estimators/objects can be found in an
    # equivalent class file in the onedal module.

    #######################
    # DOCUMENTATION NOTES #
    #######################
    #
    # All public methods (i.e. without leading underscores) should have
    # documentation which conforms to the numpy-doc standard.  Generally
    # if a defined method replaces an inherited Scikit-Learn estimator
    # method, the ``__doc__`` attribute should be re-applied to the new
    # implementation. Any new additional characteristics compared to the
    # equivalent sklearn estimator should be appended to the sklearn doc
    # string.
    #
    # When the estimator is added to the patching map in
    # sklearnex/dispatcher.py, it must be equivalently added to the support
    # table located in doc/sources/algorithms.rst if replicating an sklearn
    # estimator. If it is unique to sklearnex, it must be added to
    # docs/sources/non-scikit-algorithms.rst instead.

    # This is required as part of sklearn conformance, which does checking
    # of parameters set in __init__ when calling self.validate_params (should
    # only be in a fit or fit-derived call)
    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**_sklearn_DummyRegressor._parameter_constraints}

    def __init__(self, *, strategy="mean", constant=None, quantile=None):
        # Object instantiation is strictly limited by sklearn. It is only
        # allowed to take the keyword arguments and store them as
        # attributes with the same name. When replicating a sklearn
        # estimator, it may be possible to use the inherited version of
        # ``__init__`` from sklearn. The prototype uses defined parameters
        # to highlight the way parameters are set.  Controlled by sklearn
        # test_common.py testing.
        #
        # The signature of the __init__ must match the sklearn estimator
        # that it replicates (and is verified in test_patching.py)
        self.strategy = strategy
        self.constant = constant
        self.quantile = quantile

    # To generalize for spmd and other use cases, the constructor of the
    # onedal estimator should be set as an attribute of the class
    _onedal_DummyEstimator = staticmethod(onedal_DummyEstimator)

    ############################
    # TIER 1 METHOD FLOW NOTES #
    ############################
    #
    # Some knowledge of the process flow from the sklearnex perspective is
    # necessary to understand how to implement an estimator. For Tier 1
    # methods, the general process is as follows:
    #
    # 1) If a method which requires a fitted estimator, the method must
    # call ``check_is_fitted`` before calling ``dispatch``. This verifies
    # that aspects of the fit are available for analysis (whether oneDAL
    # may be used or not), usually this means specific instance attributes
    # with trailing underscores.
    #
    # 2) ``dispatch`` is called. This takes the estimator object, method
    # name, and the two possible evaluation branches and proceeds to call
    # ``_onedal_gpu_supported`` if a SYCL queue is found or set via the
    # target offload config. Otherwise ``_onedal_cpu_supported`` is
    # called.
    #
    # 3) ``_onedal_gpu_supported`` or ``_onedal_cpu_supported`` creates a
    # PatchingConditionsChain object, takes the input data and estimator
    # parameters, and evaluates whether the estimator and data can be run
    # using oneDAL. This information is logged to the `sklearnex` logger
    # via central code (e.g. not by the estimator) in sklearnex.
    #
    # 4) Either sklearn is called, or a object from onedal is created and
    # called using the input data. This process is handled in a function
    # which has the prefix "_onedal_" followed by the method name. When
    # fitting data, the returned onedal estimator object is stored as the
    # ``_onedal_estimator`` attribute.
    #
    # 5) Result data is returned from the estimator if necessary. Attributes
    # from the onedal estimator are copied over to the sklearnex estimator.

    def fit(self, X, y, sample_weight=None):
        # Parameter validation must be done before calls to dispatch. This
        # guarantees that the sklearn and onedal use of parameters are
        # properly typed and valued.
        if sklearn_check_version("1.2"):
            self._validate_params()

        # only arguments are passed to _onedal_*_supported, not kwargs.
        # The choice between sklearn and onedal is based off of the args,
        # and not the keyword arguments.
        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_DummyRegressor.fit,
            },
            X,
            y,
            sample_weight,
        )
        # For sklearnex-only estimators, _onedal_*_supported should either
        # pass or throw an exception. This means the sklearn branch is never
        # used. In general, the two branches must be the class methods. THe
        # parameters which are passed as arguments are given to
        # _onedal_*_supported. In this example, the ``sample_weight`` kwarg
        # is set as an arg for checking.

        # methods which do not return a result should return self (sklearn
        # standard)
        return self

    def predict(self, X, return_std=False):
        # note return_std is a special aspect of the sklearn version of this
        # estimator, normally the signatures is just predict(self, X)

        check_is_fitted(self)  # first check if fitting has occurred
        # No need to do another parameter check. While they are modifiable
        # in sklearn and in sklearnex, the parameters should never be
        # changed by hand.
        return dispatch(
            self,
            "predict",
            {
                "onedal": self.__class__._onedal_predict,
                "sklearn": _sklearn_DummyRegressor.predict,
            },
            X,
            return_std=return_std,  # not important for patching, set as kwarg
        )
        # return value will be handled by self._onedal_predict

    def _onedal_fit(self, X, y, sample_weight=None, queue=None):
        # The queue attribute must be added as the last kwarg to all
        # onedal-facing functions.  The SYCL queue is acquired in
        # ``dispatch`` and is set there before calling ``_onedal_``-prefix
        # methods.

        # The first step is to always acquire the namespace of input data
        # This is important for getting the proper data types and possibly
        # other conversions.
        xp, _ = get_namespace(X, y)

        # The second step must always be to validate the data.
        # This algorithm can accept 2d y inputs (by setting multi_output).
        # Note the use of "sklearn_check_version". This is required in order
        # to conform to changes which occur in sklearn over the supported
        # versions.  The conformance to sklearn should occur in this object,
        # therefore this function should not be used in the onedal module.
        # This conformance example is specific to the Dummy Estimators.
        X, y = validate_data(
            self, X, y, dtype=[xp.float64, xp.float32], multi_output=True, y_numeric=True, ensure_2d=sklearn_check_version("1.2")
        )
        # validate_data does several things:
        # 1) If not in the proper namespace (depending on array_api configs)
        # convert the data to the proper data format (default: numpy array)
        # 2) It will check additional aspects for labeled data.
        # 3) It will convert the arrays to the proper data type, which for
        # oneDAL is usually float64 or float32, but can also be int32 in
        # rare circumstances.
        # kwargs often are used for sklearn's ``check_array``. It is best
        # to often use the values set for sklearn for the equivalent same
        # step. This is not guaranteed and requires care by the developer.
        # For example, ``ensure_all_finite`` is set to false in this case
        # for the nature of the class, but would otherwise be unset.

        # Conformance to sklearn's DummyRegressor
        if y.ndim == 1:
            y = xp.reshape(y, (-1, 1))
        self.n_outputs_ = y.shape[1]

        # In the ``fit`` method, a Python onedal estimator object is
        # generated.
        self._onedal_estimator = self._onedal_DummyEstimator(constant=self.constant)
        # queue must be passed to the onedal Python estimator object
        # though this may change in the future as a requirement.
        self._onedal_estimator.fit(X, y, queue=queue)

        # set attributes from _onedal_estimator to sklearnex estimator
        # It is allowed to have a separate private function to do this step
        # Below is only an example, but should be all the attributes
        # available from the same sklearn estimator (if not sklearnex-only)
        # after fitting.
        self.constant_ = self._onedal_estimator.constant_
        # See sklearn conventions about trailing underscores for fitted
        # values.

        # sklearn conformance
        if self.n_outputs_ != 1 and self.constant_.shape[0] != y.shape[1]:
            raise ValueError(
                "Constant target value should have shape (%d, 1)." % y.shape[1]
            )

    def _onedal_predict(self, X, return_std=None, queue=None):
        # The first step is to always acquire the namespace of input data
        # This is important for getting the proper data types and possibly
        # other conversions.
        xp, _ = get_namespace(X)

        # The second step must always be to validate the data.
        X = validate_data(self, X, dtype=[xp.float64, xp.float32], reset=False)
        # queue must be sent back to the onedal Python estimator object
        y = self._onedal_estimator.predict(X, queue=queue)

        if self.n_outputs_ == 1:
            y = xp.reshape(y, (-1,))

        y_std = xp.zeros_like(y)

        return (y, y_std) if return_std else y

    def _onedal_cpu_supported(self, method_name, *data):
        # All estimators must have the following two functions with exactly
        # these signatures. method_name is a string which must match one
        # of the tier 1 methods of the estimator.  The logic located here
        # will inspect attributes of the data and the estimator to see if
        # sklearn

        # Begin by generating the PatchingConditionsChain, which should
        # require modifying the secondary module to match the folder as in
        # the example below.
        patching_status = PatchingConditionsChain(
            f"sklearnex.test.{self.__class__.__name__}.{method_name}"
        )
        # The conditions are specifically tailored to compares aspects
        # of the oneDAL implementation to the aspects of the sklearn
        # estimator.  For example, oneDAL may not support sparse inputs
        # where sklearn might, that would need to be checked with
        # scipy.sparse.issparse(X). In general the conditions will
        # correspond to information in the metadata and/or the estimator
        # parameters.
        #
        # In no circumstance should ``validate_data`` be called here or
        # in _onedal_gpu_supoorted to get the data into the proper form.
        if method_name == "fit":
            (X, y, sample_weight) = data
            xp, _ = get_namespace(X, y)

            # the PatchingConditionsChain is validated using
            # ``and_conditions``, use of ``or_conditions`` is highly
            # discouraged. The following checks are specific to this example
            # and must be tailored to the specific estimator implementation.
            patching_status.and_conditions(
                [
                    (
                        not sp.issparse(X),
                        "estimator set not to check input data with oneDAL",
                    ),
                    (
                        self.strategy == "constant",
                        "only the constant strategy is supported",
                    ),
                    (
                        not hasattr(X, "dtype") or X.dtype in (xp.float64, xp.float32),
                        "oneDAL operates with float64 and float32 inputs",
                    ),
                    (
                        isinstance(self.constant, (int, float)),
                        "only basic Python types are supported",
                    ),
                    (sample_weight is None, "sample_weight is not supported"),
                ]
            )

        elif method_name == "predict":
            # There is a very important subtlety about the ``dispatch`` function
            # and how it interacts with ``_onedal_*_supported`` in that only args
            # are used in these methods to evaluate oneDAL support. This means
            # that kwargs to the public API may become args in the call to dispatch
            # In this case, return_std (from predict) does not impact oneDAL, and
            # is kept as a kwarg in the ``dispatch`` call in ``predict``. In ``fit``
            # the kwarg ``sample_weight`` is important for evaluating oneDAL support
            # and is passed as an arg.
            (X,) = data
            xp, _ = get_namespace(X)

            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained."),
                    (
                        not sp.issparse(X),
                        "estimator set not to check input data with oneDAL",
                    ),
                ]
            )

        # the patching_status object should be returned
        return patching_status

    def _onedal_gpu_supported(self, method_name, *data):
        # This method will only be called if it is expected to try and use
        # a SYCL-enabled GPU. See _onedal_cpu_supported for initial
        # implementation notes.  This should follow the same procedures
        # dicatated by the characteristics of GPU oneDAL algorithm
        patching_status = PatchingConditionsChain(
            f"sklearnex.test.{self.__class__.__name__}.{method_name}"
        )
        if method_name == "fit":
            (X, y, sample_weight) = data
            xp, _ = get_namespace(X, y)

            patching_status.and_conditions(
                [
                    (
                        not sp.issparse(X),
                        "estimator set not to check input data with oneDAL",
                    ),
                    (
                        self.strategy == "constant",
                        "only the constant strategy is supported",
                    ),
                    (
                        not hasattr(X, "dtype") or X.dtype in (xp.float64, xp.float32),
                        "oneDAL operates on float64 and float32 inputs",
                    ),
                    (
                        isinstance(self.constant, (int, float)),
                        "only basic Python types are supported",
                    ),
                    (sample_weight is None, "sample_weight is not supported"),
                ]
            )

        elif method_name == "predict":
            (X,) = data
            xp, _ = get_namespace(X)

            patching_status.and_conditions(
                [
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained."),
                    (
                        not sp.issparse(X),
                        "estimator set not to check input data with oneDAL",
                    ),
                ]
            )

        # the patching_status object should be returned
        return patching_status

    # onedal estimators with onedal models which can be modified must have
    # the necessary attributes linked. This way the state of the two
    # estimators do not diverge as modifications could impact the inference
    # results. This not always necessary, as some estimators generate a
    # model for predict during fit which cannot be modified. The easiest
    # way to check for this is if the oneDAL estimator contains a "model"
    # method.

    @property
    def constant_(self):
        return self._constant_

    @constant_.setter
    def constant_(self, value):
        self._constant_ = value
        if hasattr(self, "_onedal_estimator"):
            self._onedal_estimator._onedal_model = None
            self._onedal_estimator.constant_ = value

    @constant_.deleter
    def constant_(self):
        del self._constant_

    # score is a tier 3 method in this case. Wrap with ``support_input_format`` for array
    # api support.
    score = support_input_format(_sklearn_DummyRegressor.score)

    # Docstrings should be inherited from the sklearn estimator if possible
    # In sklearnex-only estimators, they should be written from scratch
    # using the numpy-doc standard.
    __doc__ = _sklearn_DummyRegressor.__doc__
    fit.__doc__ = _sklearn_DummyRegressor.fit.__doc__
    predict.__doc__ = _sklearn_DummyRegressor.predict.__doc__
    score.__doc__ = _sklearn_DummyRegressor.score.__doc__

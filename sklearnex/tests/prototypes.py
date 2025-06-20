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
comments guiding code development should be removed unless pertinent to the
implementation.  """
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, sklearn_check_version
from onedal.tests.prototypes import PrototypeEstimator as onedal_PrototypeEstimator

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import validate_data

# if a sklearn estimator is getting replicated, it should be imported here
# with the prefix _sklearn_ added to it (using `import as`).


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
# 4) If a sklearn estimator is imported, it must have the ``_sklearn_``
# prefix added upon import in order to prevent its discovery, highlight
# its nature as private, and prevent a namespace collision. Any onedal
# imported estimator should similarly  have the ``onedal_`` prefix added
# (as it should have the same name as the sklearnex estimator).
#
# 5) ``dispatch`` is a key central function for evaluating data with either
# oneDAL or sklearn. All oneDAL algorithms which are to be directly used
# should be accessed via this function.
#
# 6) ``PatchingConditionsChain`` is used in conjunction with ``dispatch``
# and methods ``_onedal_cpu_supported`` and ``_onedal_gpu_supported`` to
# evaluate if the required evaluation on data is possible with oneDAL or
# sklearn.
#
# 7) ``get_namespace`` is key for array_api support, which yields the
# namespace associated with the given array for use in data conversion
# necessary for to and from oneDAL. An internal version is preferred due to
# limitations in sklearn versions and specific IntelPython data framework
# support (see dpctl tensors and dpnp).
#
# 8) ``validate_data`` checks data quality and estimator status before
# evaluating the function. This replicates a sklearn functionality with key
# performance changes implemented in oneDAL and therefore should only be
# imported from sklearnex and not sklearn.
#
# 9) All estimators require validation of the parameters given at
# initialization. This aspect was introduced in sklearn 1.2, any additional
# parameters must extend the dictionary for checking.

##########################
# METHOD HEIRARCHY NOTES #
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
class PrototypeEstimator(oneDALEstimator, BaseEstimator):
    # PrototypeEstimator is a sklearnex-only estimator, shown by the
    # inheritance of BaseEstimator. Inherited oneDALEstimator for estimators
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
    # All estimators should be defined in a python file located in a folder
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
    # having a python interface with oneDAL. Finally, this python object
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
    # pybind11 interface should be invoked by the python onedal estimator
    # object.
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

    def __init__(self, check=True, only_float64=False):
        # Object instantiation is strictly limited by sklearn. It is only
        # allowed to take the keyword arguments and store them as
        # attributes with the same name. When replicating a sklearn
        # estimator, it may be possible to use the inherited version of
        # ``__init__`` from sklearn. The prototype uses defined parameters
        # to highlight the process in detail when adding new parameters to
        # an estimator.

        # This estimator will abstract over the oneDAL finiteness checker
        # which usually only operates with contiguous data.  These
        # parameters will flag whether to actually check for finiteness
        # using onedal via the paramter ``check`` and check finiteness only
        # for float64 data (``only_float64``). Therefore, these two are
        # illustrative.
        self.check = check
        self.only_float64 = only_float64

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
    # may be used or not).
    #
    # 2) ``dispatch`` is called. This takes the estimator object, method
    # name, and the two possible evaluation branches and proceeds to call
    # ``_onedal_gpu_supported`` if a SYCL queue is found or set via the
    # ``target_offload`` config. Otherwise ``_onedal_cpu_supported`` is
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

    def fit(self, X, y):
        """
        Check (X, y) data for finiteness.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """

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
                "onedal": self._onedal_fit,
                "sklearn": None,  # see note below about this value
            },
            X,
            y,
        )
        # For sklearnex-only estimators, _onedal_*_supported should either
        # pass or throw an exception. This means the sklearn branch is never
        # used. Therefore the value can be set to None.  For sklearnex
        # estimators which mimic sklearn, the function should be directly
        # taken from the sklearn estimator. For example, for sklearnex's
        # DBSCAN, it will not call ``self.fit``. Instead it will use
        # ``"sklearn": sklearn_DBSCAN.fit`` directly.  This is even though
        # the estimator inherits the sklearn estimator, and can be
        # technically found via ``super``.

        # methods which do not return a result should return self (sklearn
        # standard)
        return self

    def predict(self, X):
        """Predict finiteness for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (1,)
            Float representation of finiteness.
        """
        check_is_fitted(self)  # first check if fitting has occured
        # No need to do another parameter check. While they are modifiable
        # in sklearn and in sklearnex, the parameters should never be
        # changed by hand.
        return dispatch(
            self,
            "predict",
            {
                "onedal": self._onedal_predict,
                "sklearn": None,  # see note in ``fit``
            },
            X,
        )
        # return value will be handled by self._onedal_predict

    def _onedal_fit(self, X, y, queue=None):
        # The queue attribute must be added as the last kwarg to all
        # onedal-facing functions.  The SYCL queue is acquired in
        # ``dispatch`` and is set there before calling ``_onedal_``-prefix
        # methods.

        # The first step is to always acquire the namespace of input data
        # This is important for getting the proper data types and possibly
        # other conversions.
        xp, _ = get_namespace(X, y)

        # The second step must always be to validate the data.
        X, y = validate_data(
            X, y, dtype=[xp.float64, xp.float32], ensure_all_finite=False
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

        # In the ``fit`` method, a python onedal estimator object is
        # generated.
        self._onedal_estimator = onedal_PrototypeEstimator(check=True)
        # queue must be sent back to the onedal python estimator object
        self._onedal_estimator.fit(X, y, queue=queue)

        # set attributes from _onedal_estimator to sklearnex estimator
        # It is allowed to have a separate private function to do this step
        # Below is only an example, but should be all the attributes
        # available from the same sklearn estimator (if not sklearnex-only)
        # after fitting.
        self.finite_ = self._onedal_estimator.finite_
        # See sklearn conventions about trailing underscores for fitted
        # values.

    def _onedal_predict(self, X, y, queue=None):
        # The first step is to always acquire the namespace of input data
        # This is important for getting the proper data types and possibly
        # other conversions.
        xp, _ = get_namespace(X, y)

        # The second step must always be to validate the data.
        X = validate_data(X, dtype=[xp.float64, xp.float32], ensure_all_finite=False)
        # queue must be sent back to the onedal python estimator object
        return self._onedal_estimator.predict(X, queue=queue)

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
        if method_name == "fit":
            (X, y) = data
            xp = get_namespace(X, y)

            # the PatchingConditionsChain is validated using
            # ``and_conditions``, use of ``or_conditions`` is highly
            # discouraged. The following checks are specific to this example
            # and must be tailored to the specific estimator implementation.
            patching_status.and_conditions(
                [
                    (self.check, "estimator set not to check input data with oneDAL"),
                    (
                        not self.only_float64
                        or hasattr(X, "dtype")
                        and X.dtype == xp.float64,
                        "X data is not float64 for float64-only finiteness checking",
                    ),
                    (
                        not self.only_float64
                        or hasattr(y, "dtype")
                        and y.dtype == xp.float64,
                        "y data is not float64 for float64-only finiteness checking",
                    ),
                ]
            )

        elif method_name == "predict":
            (X,) = data
            xp = get_namespace(X, y)

            patching_status.and_conditions(
                [  # a condition for ``_onedal_estimator`` is normally
                    # required if the method previously calls
                    # ``check_is_fitted``
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained."),
                    (self.check, "estimator set not to check input data with oneDAL"),
                    (
                        not self.only_float64
                        or hasattr(X, "dtype")
                        and X.dtype == xp.float64,
                        "X data is not float64 for float64-only finiteness checking",
                    ),
                ]
            )

        # the patching_status object should be returned
        return patching_status

    def _onedal_gpu_supported(self, method_name, *data):
        # This method will only be called if it is expected to try and use
        # a SYCL-enabled GPU. See _onedal_cpu_supported for initial
        # implemenatation notes.  This should follow the same proceedures
        # dicatated by the characteristics of GPU oneDAL algorithm
        patching_status = PatchingConditionsChain(
            f"sklearnex.test.{self.__class__.__name__}.{method_name}"
        )
        if method_name == "fit":
            (X, y) = data
            xp = get_namespace(X, y)

            # the PatchingConditionsChain is validated using
            # ``and_conditions``, use of ``or_conditions`` is highly
            # discouraged. The following checks are specific to this example
            # and must be tailored to the specific estimator implementation.
            patching_status.and_conditions(
                [
                    (self.check, "estimator set not to check input data with oneDAL"),
                    (
                        not self.only_float64
                        or hasattr(X, "dtype")
                        and X.dtype == xp.float64,
                        "X data is not float64 for float64-only finiteness checking",
                    ),
                    (
                        not self.only_float64
                        or hasattr(y, "dtype")
                        and y.dtype == xp.float64,
                        "y data is not float64 for float64-only finiteness checking",
                    ),
                ]
            )

        elif method_name == "predict":
            (X,) = data
            xp = get_namespace(X, y)

            patching_status.and_conditions(
                [  # a condition for ``_onedal_estimator`` is normally
                    # required if the method previously calls
                    # ``check_is_fitted``
                    (hasattr(self, "_onedal_estimator"), "oneDAL model was not trained."),
                    (self.check, "estimator set not to check input data with oneDAL"),
                    (
                        not self.only_float64
                        or hasattr(X, "dtype")
                        and X.dtype == xp.float64,
                        "X data is not float64 for float64-only finiteness checking",
                    ),
                ]
            )

        # the patching_status object should be returned
        return patching_status

    def score(self, X, y):
        """
        Return float value difference in finiteness.

        This is a simple mathematical representation that should never be
        used and is simply an example.  When implementing score for any other
        estimator, it should use a score method matching the inherited mixins
        (like r2_score, accuracy_score, etc.).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples 'X'.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Values 'y'.

        Returns
        -------
        score : float
            difference in finiteness as a float
        """
        # This is an example tier 2 method which uses some small additional
        # functionality on top of a tier 1 method
        # This return value here is a trivial example.
        return self.predict(X) - self.predict(y)

    # These are commented out, as they are generally necessary for copying
    # docstrings from sklearn estimators. _sklearn_Estimator_ should be
    # re-named. For example, for PCA, the inherited sklearn estimator should
    # be _sklearn_PCA.
    # fit.__doc__ = _sklearn_Estimator.__doc__
    # predict.__doc__ = _sklearn_Estimator.__doc__
    # score.__doc__ = _sklearn_Estimator.__doc__

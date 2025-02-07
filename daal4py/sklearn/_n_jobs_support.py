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

import logging
import sys
import threading
from functools import wraps
from inspect import Parameter, signature
from joblib import cpu_count
from numbers import Integral
from warnings import warn

import threadpoolctl

from daal4py import _get__daal_link_version__, daalinit, num_threads

from ._utils import sklearn_check_version

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import validate_parameter_constraints


class oneDALLibController(threadpoolctl.LibController):
    user_api="oneDAL"
    internal_api="oneDAL"
    
    filename_prefixes = ("libonedal_thread", "libonedal")
    
    def get_num_threads(self):
        return num_threads()

    def set_num_threads(self, nthreads):
        daalinit(nthreads)

    def get_version(self):
        return _get__daal_link_version__

threadpoolctl.register(oneDALLibController)

# Note: getting controller in global scope of this module is required
# to avoid overheads by its initialization per each function call
threadpool_controller = threadpoolctl.ThreadpoolController()
# similarly the number of cpus is not expected to change after import
_cpu_count = cpu_count()

def _run_with_n_jobs(method):
    """
    Decorator for running of methods containing oneDAL kernels with 'n_jobs'.

    Outside actual call of decorated method, this decorator:
    - checks correctness of passed 'n_jobs',
    - deducts actual number of threads to use,
    - sets and resets this number for oneDAL environment.
    """

    @wraps(method)
    def n_jobs_wrapper(self, *args, **kwargs):
        # threading parallel backend branch
        # multiprocess parallel backends branch
        # preemptive validation of n_jobs parameter is required
        # because '_run_with_n_jobs' decorator is applied on top of method
        # where validation takes place
        if sklearn_check_version("1.2") and hasattr(self, "_parameter_constraints"):
            validate_parameter_constraints(
                parameter_constraints={"n_jobs": self._parameter_constraints["n_jobs"]},
                params={"n_jobs": self.n_jobs},
                caller_name=self.__class__.__name__,
            )

        # receive n_threads limitation from upper parallelism context
        # using `threadpoolctl.ThreadpoolController`
        # get real `n_jobs` number of threads for oneDAL
        # using sklearn rules and `n_threads` from upper parallelism context

        if not self.n_jobs:
            n_jobs = _cpu_count
        else:
            n_jobs = self.n_jobs if self.n_jobs > 0 else max(1, _cpu_count + self.n_jobs + 1)

        if (old_n_threads := threadpool_controller.lib_controllers['oneDAL'].num_threads) != n_jobs:
            logger = logging.getLogger("sklearnex")
            cl = self.__class__
            logger.debug(
                f"{cl.__module__}.{cl.__name__}.{method.__name__}: "
                f"setting {n_jobs} threads (previous - {old_n_threads})"
            )
            with threadpool_controller.limit(limits=n_jobs, user_api='oneDAL'):
                return method(self, *args, **kwargs)
        else:
            return method(self, *args, **kwargs)
    return n_jobs_wrapper


def control_n_jobs(decorated_methods: list = []):
    """
    Decorator for controlling the 'n_jobs' parameter in an estimator class.

    This decorator is designed to be applied to both estimators with and without
    native support for the 'n_jobs' parameter in the original Scikit-learn APIs.
    When applied to an estimator without 'n_jobs' support in
    its original '__init__' method, this decorator adds the 'n_jobs' parameter.

    Additionally, this decorator allows for fine-grained control over which methods
    should be executed with the 'n_jobs' parameter. The methods specified in
    the 'decorated_methods' argument will run with 'n_jobs',
    while all other methods remain unaffected.

    Parameters
    ----------
    decorated_methods: list
        A list of method names to be executed with 'n_jobs'.

    Example
    -------
        @control_n_jobs(decorated_methods=['fit', 'predict'])

        class MyEstimator:

            def __init__(self, *args, **kwargs):
                # Your original __init__ implementation here

            def fit(self, *args, **kwargs):
                # Your original fit implementation here

            def predict(self, *args, **kwargs):
                # Your original predict implementation here

            def other_method(self, *args, **kwargs):
                # Methods not listed in decorated_methods will not be affected by 'n_jobs'
                pass
    """

    def class_wrapper(original_class):
        original_class._n_jobs_supported_onedal_methods = decorated_methods.copy()

        original_init = original_class.__init__

        if sklearn_check_version("1.2") and hasattr(
            original_class, "_parameter_constraints"
        ):
            parameter_constraints = original_class._parameter_constraints
            if "n_jobs" not in parameter_constraints:
                parameter_constraints["n_jobs"] = [Integral, None]

        @wraps(original_init)
        def init_with_n_jobs(self, *args, n_jobs=None, **kwargs):
            original_init(self, *args, **kwargs)
            self.n_jobs = n_jobs

        # add "n_jobs" parameter to signature of wrapped init
        # if estimator doesn't originally support it
        sig = signature(original_init)
        if "n_jobs" not in sig.parameters:
            params_copy = sig.parameters.copy()
            params_copy.update(
                {
                    "n_jobs": Parameter(
                        name="n_jobs", kind=Parameter.KEYWORD_ONLY, default=None
                    )
                }
            )
            init_with_n_jobs.__signature__ = sig.replace(parameters=params_copy.values())
            original_class.__init__ = init_with_n_jobs

        # add n_jobs to __doc__ string if needed
        if (
            hasattr(original_class, "__doc__")
            and isinstance(original_class.__doc__, str)
            and "n_jobs : int" not in original_class.__doc__
        ):
            # Python 3.13 removed extra tab in class doc string
            tab = "    " if sys.version_info.minor < 13 else ""
            parameters_doc_tail = f"\n{tab}Attributes"
            n_jobs_doc = f"""
{tab}n_jobs : int, default=None
{tab}    The number of jobs to use in parallel for the computation.
{tab}    ``None`` means using all physical cores
{tab}    unless in a :obj:`joblib.parallel_backend` context.
{tab}    ``-1`` means using all logical cores.
{tab}    See :term:`Glossary <n_jobs>` for more details.
"""
            original_class.__doc__ = original_class.__doc__.replace(
                parameters_doc_tail, n_jobs_doc + parameters_doc_tail
            )

        # decorate methods to be run with applied n_jobs parameter
        for method_name in decorated_methods:
            # if method doesn't exist, we want it to raise an Exception
            method = getattr(original_class, method_name)
            if not hasattr(method, "__onedal_n_jobs_decorated__"):
                decorated_method = _run_with_n_jobs(method)
                # sign decorated method for testing and other purposes
                decorated_method.__onedal_n_jobs_decorated__ = True
                setattr(original_class, method_name, decorated_method)
            else:
                warn(
                    f"{original_class.__name__}.{method_name} already has "
                    "oneDAL n_jobs support and will not be decorated."
                )

        return original_class

    return class_wrapper

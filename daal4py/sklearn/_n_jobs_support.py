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
from multiprocessing import cpu_count
from numbers import Integral
from warnings import warn

import threadpoolctl

from daal4py import daalinit as set_n_threads
from daal4py import num_threads as get_n_threads

from ._utils import sklearn_check_version

if sklearn_check_version("1.2"):
    from sklearn.utils._param_validation import validate_parameter_constraints


# Note: getting controller in global scope of this module is required
# to avoid overheads by its initialization per each function call
threadpool_controller = threadpoolctl.ThreadpoolController()


def get_suggested_n_threads(n_cpus):
    """
    Function to get `n_threads` limit
    if `n_jobs` is set in upper parallelization context.
    Usually, limit is equal to `n_logical_cpus` // `n_jobs`.
    Returns None if limit is not set.
    """
    n_threads_map = {
        lib_ctl.internal_api: lib_ctl.get_num_threads()
        for lib_ctl in threadpool_controller.lib_controllers
        if lib_ctl.internal_api != "mkl"
    }
    # openBLAS is limited to 24, 64 or 128 threads by default
    # depending on SW/HW configuration.
    # thus, these numbers of threads from openBLAS are uninformative
    if "openblas" in n_threads_map and n_threads_map["openblas"] in [24, 64, 128]:
        del n_threads_map["openblas"]
    # remove default values equal to n_cpus as uninformative
    for backend in list(n_threads_map.keys()):
        if n_threads_map[backend] == n_cpus:
            del n_threads_map[backend]
    if len(n_threads_map) > 0:
        return min(n_threads_map.values())
    else:
        return None


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
        if not isinstance(threading.current_thread(), threading._MainThread):
            warn(
                "'Threading' parallel backend is not supported by "
                "Extension for Scikit-learn*. "
                "Falling back to usage of all available threads."
            )
            result = method(self, *args, **kwargs)
            return result
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
        # search for specified n_jobs
        n_jobs = self.n_jobs
        n_cpus = cpu_count()
        # receive n_threads limitation from upper parallelism context
        # using `threadpoolctl.ThreadpoolController`
        n_threads = get_suggested_n_threads(n_cpus)
        # get real `n_jobs` number of threads for oneDAL
        # using sklearn rules and `n_threads` from upper parallelism context
        if n_jobs is None or n_jobs == 0:
            if n_threads is None:
                # default branch with no setting for n_jobs
                return method(self, *args, **kwargs)
            else:
                n_jobs = n_threads
        elif n_jobs < 0:
            if n_threads is None:
                n_jobs = max(1, n_cpus + n_jobs + 1)
            else:
                n_jobs = max(1, n_threads + n_jobs + 1)
        # branch with set n_jobs
        old_n_threads = get_n_threads()
        if n_jobs == old_n_threads:
            return method(self, *args, **kwargs)

        try:
            logger = logging.getLogger("sklearnex")
            cl = self.__class__
            logger.debug(
                f"{cl.__module__}.{cl.__name__}.{method.__name__}: "
                f"setting {n_jobs} threads (previous - {old_n_threads})"
            )
            set_n_threads(n_jobs)
            return method(self, *args, **kwargs)
        finally:
            set_n_threads(old_n_threads)

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
            tab = (
                "    "
                if (sys.version_info.major == 3 and sys.version_info.minor < 13)
                else ""
            )
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

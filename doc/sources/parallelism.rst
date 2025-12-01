.. Copyright contributors to the oneDAL project

.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at

..      http://www.apache.org/licenses/LICENSE-2.0

.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. include:: substitutions.rst

#####################
Parallelism Specifics
#####################

|sklearnex| supports the :term:`n_jobs` parameter of the original |sklearn| with the following differences:

* `n_jobs` parameter is supported for all estimators patched by |sklearnex|,
  while |sklearn| enables it for selected estimators only.
* `n_jobs` estimator parameter sets the number of threads used by the underlying |onedal|.
* |sklearnex| doesn't use :mod:`joblib` for parallelism in patched estimators and functions.
* The only low-level parallelism library used by |sklearnex| is `oneTBB <https://github.com/uxlfoundation/oneTBB>`__ 
  (through the |onedal| and `oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`__).
* If `n_jobs` is not specified, |sklearnex| uses all available threads whereas |sklearn| is single-threaded by default.
  Note that the deprecated :doc:`daal4py <daal4py>` module uses a global configuration instead of per-object ``n_jobs`` arguments, 
  with the default also being all available threads.

|sklearnex| follows the same rules as |sklearn| for
the calculation of the :term:`n_jobs` parameter value.

When |sklearn|'s utilities with built-in parallelism are used 
(for example, :obj:`sklearn.model_selection.GridSearchCV` or :obj:`sklearn.model_selection.VotingClassifier`),
|sklearnex| tries to determine the optimal number of threads per job using hints provided by :mod:`joblib` / ``threadpoolctl``..
If ``n_jobs`` is not specified for underlying estimator(s), |sklearnex| sets it to the number of available threads
(usually the number of logical CPUs divided by `n_jobs` set for higher-level parallelized entities).

.. note::
    Environment variables such as `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and others used by
    low-level parallelism libraries do not affect |sklearnex|, nor does the 
    `mkl-service <https://github.com/IntelPython/mkl-service>`__ package.

.. note::
    ``n_jobs`` has no effect if computations are performed on GPU.

.. note::
    `threadpoolctl` context has no effect on |sklearnex| threading if `n_jobs` is specified and non-negative.
    If `n_jobs` is equal to `0` or not specified then the number from `threadpoolctl` is propagated to |sklearnex|.
    If `n_jobs` is negative then the `threadpoolctl`'s number will be `max(1, n_threadpoolctl + n_jobs + 1)`.

.. note::
    |sklearnex| threading doesn't automatically avoid nested parallelism when used in conjunction with OpenMP and/or python threads.

.. warning::
    If several instances of |sklearnex| algorithms are run sequentially and the ``n_jobs`` parameter for the first run
    is significantly greater than for subsequent ones, it may result in performance degradation due to a known issue
    with `oneTBB <https://github.com/uxlfoundation/oneTBB>`__.

.. warning::
    In general, accelerated computations offered by estimators from the |sklearnex|
    do not raise the Python GIL, thus they are not compatible with multi-threading
    backends that rely on Python threads.

.. warning::
    Internally, the number of threads for calls to estimator methods from
    the |sklearnex| is managed through global variables - thus, if multiple
    calls to estimators with different ``n_jobs`` are performed in parallel
    through **Python threads**, there might be threading races that override
    one another's configuration, potentially leading to process-wide crashes.
    If concurrent calls are to be performed, process-based parallelism should
    be used instead.

Setting the `DEBUG` :ref:`verbosity setting <verbose>` will produce logs
indicating when the number of threads used is different from the default
(number of logical threads in the machine).

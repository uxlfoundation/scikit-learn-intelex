.. Copyright 2025 Intel Corporation
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.

.. _parallelism:

#####################
Parallelism Specifics
#####################

|sklearnex| supports the `n_jobs <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_ parameter
of the original |sklearn| with the following differences:

* `n_jobs` parameter is supported for all estimators patched by |sklearnex|,
  while |sklearn| enables it for selected estimators only
* `n_jobs` estimator parameter sets the number of threads used by the underlying |oneDAL|
* |sklearnex| doesn't use `joblib` for parallelism in patched estimators and functions
* The only low-level parallelism library used by |sklearnex| is oneTBB (through oneDAL)
* The `threading` parallel backend of `joblib` is not supported by |sklearnex|

The only exception is multiclass LogisticRegression, which uses `joblib` for parallelism across classes.

|sklearnex| follows the same rules as |sklearn| for
`the calculation of the 'n_jobs' parameter value <https://scikit-learn.org/stable/glossary.html#term-n_jobs>`_.

When Scikit-learn's utilities with built-in parallelism are used (for example, `GridSearchCV` or `VotingClassifier`),
|sklearnex| tries to determine the optimal number of threads per job using hints provided by `joblib`.
If `n_jobs` is not specified for underlying estimator(s), |sklearnex| sets it to the number of available threads
(usually the number of logical CPUs divided by `n_jobs` set for higher-level parallelized entities).

Environment variables such as `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and others used by
low-level parallelism libraries are recognized by `joblib` and therefore can be used as hints by |sklearnex|.

To track the actual number of threads used by sklearnex's estimators,
set the `DEBUG` :ref:`verbosity setting <_verbose>`.

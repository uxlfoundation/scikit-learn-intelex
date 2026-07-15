.. Copyright contributors to the oneDAL project
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

.. include:: substitutions.rst

##############################
Unsupported |sklearn| features
##############################

In general, estimators and functions from the |sklearnex| are sub-classed from their analogs in |sklearn| and are fully API compatible, but some particular features offered by |sklearn| cannot be used in the |sklearnex| due to differences in how the two libraries work internally.

Python threads
--------------

Estimators and functions from the |sklearnex| are not meant to be parallelizable through Python threads, since they do not raise the GIL (global interpreter lock) and in some cases might rely on global variables. Note that, at this moment, the |sklearnex| does not provide builds for free-threaded Python that could be used without the GIL.

Process-based parallelization (the default mode in :mod:`joblib`) is recommended instead, whether through forked processes or through sub-processes. The |sklearnex| is fork-safe and this parallelization mode will usually be faster on Linux*, but note that other libraries that could be used together with it (such as `Polars <https://pola.rs>`__) might not always work correctly under forked processes.

See :doc:`parallelism` for more details.

.. _config_unsupported:

Configurable options
--------------------

In |sklearn|, several options can be managed through :obj:`sklearn.config_context` and :obj:`sklearn.set_config`. These are propagated to estimators and functions from the |sklearnex| under both the stock and the patched versions of these functions (see :doc:`config-contexts`), but when accelerated routines are used, the following options will not have any effect:

- ``working_memory``.
- ``pairwise_dist_chunk_size``.
- ``enable_cython_pairwise_dist``.
- ``skip_parameter_validation``.

Verbosity
---------

Some estimators in |sklearn| offer a ``verbose`` argument, such as :obj:`sklearn.ensemble.RandomForestClassifier`, which allows printing messages during model fitting to monitor how the procedure is progressing.

Verbose mode is not supported in estimators from the |sklearnex|, but different levels of verbosity can be enabled in the underlying |onedal| - see :external+onedal:doc:`contribution/profiling` for details. Note however that the messages printed by |sklearn| and by the |onedal| will not contain the same kind of information.

If argument ``verbose`` is passed to estimators in the |sklearnex| and accelerated computations on them are supported (see :doc:`algorithms`), the argument will be ignored - i.e. messages will not be printed.

Callbacks
---------

Callback functions (an experimental feature introduced in version 1.9 of |sklearn|) are not supported in estimators from the |sklearnex|. If supplied, they will not be used.

Moving estimators
-----------------

Function ``sklearn.utils._array_api.move_estimator_to`` is currently not supported for estimator objects from the |sklearnex|. See :doc:`array_api` for more details.

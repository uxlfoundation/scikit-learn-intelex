.. Copyright 2024 Intel Corporation
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
.. _preview:

#####################
Preview Functionality
#####################

Some of implemented functionality in |sklearnex| doesn't meet one or few of next requirements
for being enabled by default for all users:

* The functionality misses performance targets compared to stock |sklearn| or to previously available versions.
* The functionality API is not stable and may change in future.
* The functionality doesn't have full compatibility with its analog from |sklearn|.
* The functionality is not fully tested.

This type of functionality is available under **preview mode** of |sklearnex| and located in
submodule ``sklearnex.preview``.

Functionalities under preview will be made available after patching when preview mode is enabled,
but note that some might be :ref:`extension estimators <extension_estimators>` without analogs in |sklearn|.

Estimators in preview can be enabled by either setting environment variable ``SKLEARNEX_PREVIEW``
before patching |sklearn|, or by passing argument ``preview=True`` to function ``patch_sklearn``.

For example, the environment variable can be set from a terminal before starting python:

- On Linux* OS ::

     export SKLEARNEX_PREVIEW=1

- On Windows* OS ::

     set SKLEARNEX_PREVIEW=1

Or it can be set within a Python process:

.. code-block:: python

    import os
    os.environ["SKLEARNEX_PREVIEW"] = "1"

Then, patched estimators in preview mode can be imported from the ``slearn`` module
when they participate in patching:

.. code-block:: python

     from sklearnex import patch_sklearn
     patch_sklearn()
     from sklearn.decomposition import IncrementalPCA
     print(IncrementalPCA.__module__)
     # output:
     # sklearnex.preview.decomposition.incremental_pca

Alternatively, estimators can be imported directly from ``sklearnex.preview`` without
patching and without setting environment variable ``SKLEARNEX_PREVIEW``:

.. code-block:: python

    from sklearnex.preview.covariance import EmpiricalCovariance


Current list of preview estimators:

.. list-table::
   :widths: 30 20 10
   :header-rows: 1
   :align: left

   * - Estimator name
     - Module
     - Is patching supported
   * - :obj:`sklearn.covariance.EmpiricalCovariance`
     - ``sklearnex.preview.covariance``
     - Yes
   * - :obj:`sklearn.decomposition.IncrementalPCA`
     - ``sklearnex.preview.decomposition``
     - Yes

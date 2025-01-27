.. ******************************************************************************
.. * Copyright 2024 Intel Corporation
.. *
.. * Licensed under the Apache License, Version 2.0 (the "License");
.. * you may not use this file except in compliance with the License.
.. * You may obtain a copy of the License at
.. *
.. *     http://www.apache.org/licenses/LICENSE-2.0
.. *
.. * Unless required by applicable law or agreed to in writing, software
.. * distributed under the License is distributed on an "AS IS" BASIS,
.. * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. * See the License for the specific language governing permissions and
.. * limitations under the License.
.. *******************************************************************************/

.. _preview:

#####################
Preview Functionality
#####################

Some of implemented functionality in |intelex| doesn't meet one or few of next requirements
for being enabled by default for all users:

* The functionality API is not stable and can be changed in future
* The functionality doesn't have full compatibility with stock Scikit-learn
* The functionality misses performance targets compared to stock Scikit-learn or previously available version of functionality
* The functionality is not fully tested

This type of functionality is available under **preview mode** of |intelex| and located in
the corresponding module (`sklearnex.preview`).

Preview functionality *may* or *may not* participate in patching of Scikit-learn.
For example, a preview estimator may be a replacement for a stock one or a completely new one.

To enable preview functionality, you need to set the `SKLEARNEX_PREVIEW` environment variable
to non-empty value before patching of Scikit-learn.
For example, you can set the environment variable in the following way:

- On Linux* OS ::

     export SKLEARNEX_PREVIEW=1

- On Windows* OS ::

     set SKLEARNEX_PREVIEW=1

Then, you can import Scikit-learn estimator patched with a preview one from `sklearnex.preview` module::

     from sklearnex import patch_sklearn
     patch_sklearn()
     from sklearn.decomposition import IncrementalPCA
     print(IncrementalPCA.__module__)
     # output:
     # sklearnex.preview.decomposition.incremental_pca

Current list of preview estimators:

.. list-table::
   :widths: 30 20 10
   :header-rows: 1
   :align: left

   * - Estimator name
     - Module
     - Is patching supported
   * - EmpiricalCovariance
     - sklearnex.preview.covariance
     - Yes
   * - IncrementalPCA
     - sklearnex.preview.decomposition
     - Yes

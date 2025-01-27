.. ******************************************************************************
.. * Copyright 2020 Intel Corporation
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

.. |intelex_repo| replace:: |intelex| repository
.. _intelex_repo: https://github.com/intel/scikit-learn-intelex

.. _index:

#########
|intelex|
#########

With |intelex| you can accelerate your Scikit-learn applications and
still have full conformance with all Scikit-Learn APIs and algorithms.
|intelex| is a free software AI accelerator that brings over 10-100X acceleration across a variety of applications.

|intelex| offers you a way to accelerate existing scikit-learn code.
The acceleration is achieved through :term:`patching`:
replacing the stock scikit-learn algorithms with their optimized versions provided by the extension.

.. rubric:: Designed for Data Scientists and Framework Designers

|intelex| was created to provide data scientists with a way to get a better performance
while using the familiar scikit-learn package and getting the same results.

Usage
------

.. include:: usage.rst

Important Links
--------------------
- |intelex_repo|_
- `Machine Learning Benchmarks <https://github.com/IntelPython/scikit-learn_bench>`_
- `About Intel(R) oneAPI Data Analytics Library <https://github.com/oneapi-src/oneDAL>`_
- `About daal4py <https://github.com/intel/scikit-learn-intelex/tree/master/daal4py>`_

.. toctree::
   :caption: About
   :hidden:

   acceleration.rst
   what-is-patching.rst
   Medium Blogs <blogs.rst>
   System Requirements <system-requirements.rst>
   memory-requirements.rst

.. toctree::
   :caption: Get Started
   :hidden:

   installation.rst
   quick-start.rst
   samples.rst
   kaggle.rst

.. toctree::
   :caption: Developer Guide
   :hidden:

   algorithms.rst
   guide/acceleration.rst
   global-patching.rst
   verbose.rst
   distributed-mode.rst
   oneAPI and GPU support <oneapi-gpu.rst>

.. toctree::
   :caption: Community
   :hidden:

   support.rst
   contribute.rst

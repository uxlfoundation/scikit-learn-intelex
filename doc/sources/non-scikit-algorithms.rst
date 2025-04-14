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

.. _extension_estimators:

Non-Scikit-Learn Algorithms
===========================
Algorithms not presented in the original scikit-learn are described here. All algorithms are
available for both CPU and GPU (including distributed mode)

.. autoclass:: sklearnex.basic_statistics.BasicStatistics
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: sklearnex.basic_statistics.IncrementalBasicStatistics
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: sklearnex.covariance.IncrementalEmpiricalCovariance
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: sklearnex.linear_model.IncrementalLinearRegression
    :members:
    :inherited-members:
    :show-inheritance:

.. autoclass:: sklearnex.linear_model.IncrementalRidge
    :members:
    :inherited-members:
    :show-inheritance:

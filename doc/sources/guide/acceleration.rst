.. Copyright 2022 Intel Corporation
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

########################
Tuning Guide
########################

The performance of some algorithms changes based on the parameters that are used.
This section denotes the details of such cases.

Refer to :ref:`sklearn_algorithms` to see the full list of algorithms, parameters, and data formats supported in |sklearnex|.

.. _acceleration_tsne:

TSNE
----

TSNE algorithm consists of two components: KNN and Gradient Descent.
The overall acceleration of TSNE depends on the acceleration of each of these algorithms.

- The KNN part of the algorithm supports all parameters except:

  - ``metric`` != `'euclidean'` or `'minkowski'` with ``p`` != `2`
- The Gradient Descent part of the algorithm supports all parameters except:

  - ``n_components`` > ``2``
  - ``method`` = ``'exact'``
  - ``verbose`` != ``0``

To get better performance, use parameters supported by both components.

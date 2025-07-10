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

  - ``n_components`` = `3`
  - ``method`` = `'exact'`
  - ``verbose`` != `0`

To get better performance, use parameters supported by both components.

.. _acceleration_rf:

Random Forest
-------------

Random Forest models accelerated with |sklearnex| and using the `hist` splitting
method discretize training data by creating a histogram with a configurable
number of bins. The following keyword arguments can be used to influence the
created histogram.

.. list-table::
   :widths: 10 10 10 30
   :header-rows: 1
   :align: left

   * - Keyword argument
     - Possible values
     - Default value
     - Description
   * - ``maxBins``
     - `[0, inf)`
     - ``256``
     - Number of bins in the histogram with the discretized training data. The
       value ``0`` disables data discretization.
   * - ``minBinSize``
     - `[1, inf)`
     - ``5``
     - Minimum number of training data points in each bin after discretization.
   * - ``binningStrategy``
     - ``quantiles, averages``
     - ``quantiles``
     - Selects the algorithm used to calculate bin edges. ``quantiles``
       results in bins with a similar amount of training data points. ``averages``
       divides the range of values observed in the training data set into
       equal-width bins of size `(max - min) / maxBins`.

Note that using discretized training data can greatly accelerate model training
times, especially for larger data sets. However, due to the reduced fidelity of
the data, the resulting model can present worse performance metrics compared to
a model trained on the original data. In such cases, the number of bins can be
increased with the ``maxBins`` parameter, or binning can be disabled entirely by
setting ``maxBins=0``.

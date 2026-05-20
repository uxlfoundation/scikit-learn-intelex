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

#################################################
Additional parameters in estimators and functions
#################################################

For the most part, estimators and functions in the |sklearnex| that have an analog in |sklearn| offer the same signatures for class constructors and functions, but there are a few exceptions where the classes/functions from the |sklearnex| allow additional parameters, which will be available both under patching and when importing them from the ``sklearnex`` module.

The cases with additional parameters are listed below:

Random Forests
==============

Random Forest models (including their "Extremely Randomized" variants) accelerated with |sklearnex| use histogram-based algorithms for splitting subsamples of data, which differs a bit from the sorting-based splitting logic used in the same classes from |sklearn|. The following keyword arguments can be used to control how histograms are created:

.. list-table::
   :widths: 10 10 10 30
   :header-rows: 1
   :align: left

   * - Keyword argument
     - Possible values
     - Default value
     - Description
   * - ``max_bins``
     - `[2, inf)`
     - ``256``
     - Number of bins in the histogram with the discretized training data.
   * - ``min_bin_size``
     - `[1, inf)`
     - ``5``
     - Minimum number of training data points in each bin after discretization.

Note that using discretized training data can greatly accelerate model training times, especially for larger data sets. However, due to the reduced fidelity of the data, the resulting model can present worse performance metrics compared to a model trained on the original data. In such cases, the number of bins can be increased with the ``max_bins`` parameter.

This parameter is available in the following classes:

- :obj:`sklearn.ensemble.RandomForestRegressor`
- :obj:`sklearn.ensemble.RandomForestClassifier`
- :obj:`sklearn.ensemble.ExtraTreesRegressor`
- :obj:`sklearn.ensemble.ExtraTreesClassifier`

Train-test splitting
====================

Function :obj:`sklearn.model_selection.train_test_split` offers an additional keyword-only argument ``rng`` which can be used to select the algorithm to be used for random number generation, by passing its name as a string, with a default value of ``"OPTIMIZED_MT19937"``.

This parameter is only used when passing ``shuffle=True`` and ``stratify=None``.

If the `mkl_random <https://github.com/IntelPython/mkl_random>`__ package is installed, under the above conditions, it will be used to generate random numbers for the splits, and the ``rng`` keyword will be forwarded to ``mkl_random`` as ``brng`` argument. See the `mkl_random documentation <https://intelpython.github.io/mkl_random/reference/index.html>`__ for details about which values are allowed.

Otherwise, if passing ``rng="OPTIMIZED_MT19937"`` (the default), random numbers will be generated using an optimized version of the MT19937 algorithm (as offered by NumPy, for example) from the |onedal|.

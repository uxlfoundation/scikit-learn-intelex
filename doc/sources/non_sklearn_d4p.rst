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

.. _non_sklearn_d4p:

Other Non-Scikit-Learn Algorithms
=================================

In addition to the :ref:`extension estimators <extension_estimators>` in module
``sklearnex``, module ``daal4py`` offers Python bindings over some additional
machine learning algorithms in the |onedal| that have not been implemented by
|sklearn|, albeit through a lower-level interface than ``sklearnex`` which does not
follow the conventions from |sklearn|.

See :ref:`about_daal4py` for more details about this module.

List of additional machine learning algorithms offered through ``daal4py``:

- :obj:`Association rules <daal4py.association_rules>`
- :obj:`Implicit ALS <daal4py.implicit_als_training>`
- :obj:`BACON Outlier Detection <daal4py.bacon_outlier_detection>`
- :obj:`Stump Regression <daal4py.stump_regression_training>`
- :obj:`BrownBoost classification <daal4py.brownboost_training>`

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

==================
Patching Utilities
==================

Overview
========

The easiest way to call optimized routines from the |sklearnex| is by patching the ``sklearn`` module
from |sklearn|, which can be achieved with just a call to :obj:`sklearnex.patch_sklearn`:

.. code-block:: python

  from sklearnex import patch_sklearn
  from sklearnex import is_patched_instance
  from sklearn.linear_model import LinearRegression

  is_patched_instance(LinearRegression()) == False

  patch_sklearn()
  from sklearn.linear_model import LinearRegression # now calls sklearnex
  is_patched_instance(LinearRegression()) == True

Patching can be applied either temporarily for a Python session (change is applied at the process level),
or permanently at the module level by modifying the files in the installed ``sklearn`` module to apply
patching before importing ``sklearn``.

Either patching mechanism (process-level or module-level) can be applied from inside a Python session, or
through command line arguments for the Python process.

Temporary Patching
==================

Once you install the |sklearnex|, you can replace estimator classes (algorithms) that exist in the ``sklearn``
module from |sklearn| with their optimized versions from the extension. This action is called *patching* or
*monkey patching*. This is not a permanent change so you can always undo the patching dynamically if necessary.

To patch |sklearn| with the |sklearnex|, the following methods can be used:

.. list-table::
   :header-rows: 1
   :align: left

   * - Method
     - Action
   * - Use a flag in the command line
     - Run this command:

       ::

          python -m sklearnex my_application.py
   * - Modify your script
     - Add the following lines **before importing** from ``sklearn``:

       ::

          from sklearnex import patch_sklearn
          patch_sklearn()
   * - Import an estimator from the ``sklearnex`` module
     - Run this command:

       ::

          from sklearnex.neighbors import NearestNeighbors


These patching methods are interchangeable.

Unpatching
----------

To undo the patch (also called *unpatching*) means to return the ``sklearn`` module to the original
implementation from |sklearn|, replacing patched estimators from the |sklearnex| with their stock
|sklearn| analogs.

In order for changes to take effect, you must reimport the ``sklearn`` module(s) afterwards:

.. code-block:: python

  sklearnex.unpatch_sklearn()
  # Re-import scikit-learn algorithms after the unpatch
  from sklearn.cluster import KMeans

Example
-------

This example shows how to patch |sklearn| inside a Python session by modifying your script. To make sure that
patching is registered by the scikit-learn estimators, always import module ``sklearn`` after calling the
patching function:

.. code-block:: python
  :caption: Example: Drop-In Patching

    import numpy as np
    from sklearnex import patch_sklearn
    patch_sklearn()

    # You need to re-import scikit-learn algorithms after the patch
    from sklearn.cluster import KMeans

    # The optimized estimators follow the same API as the originals
    X = np.array([[1,  2], [1,  4], [1,  0],
                  [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    print(f"kmeans.labels_ = {kmeans.labels_}")


Permanent Patching
==================

You can also use global patching to permanently patch all your |sklearn| applications without any additional actions,
by modifying the files in the installed ``sklearn`` module.

Before you begin, make sure that you have read and write permissions for the installed ``sklearn`` module files.

With global patching, you can:

.. list-table::
   :header-rows: 1
   :align: left

   * - Task
     - Action
     - Note
   * - Patch all supported algorithms
     - Run this command:

       ::

          python -m sklearnex.glob patch_sklearn

     - If you run the global patching command several times with different parameters, then only the last configuration is applied.
   * - Patch selected algorithms
     - Use ``--algorithm`` or ``-a`` keys with a list of algorithms to patch. For example, to patch only ``SVC`` and ``RandomForestClassifier`` estimators, run

       ::

           python -m sklearnex.glob patch_sklearn -a svc random_forest_classifier

     -
   * - Enable global patching via code
     - Use the ``patch_sklearn`` function with the ``global_patch`` argument:

       ::

          from sklearnex import patch_sklearn
          patch_sklearn(global_patch=True)
          import sklearn

     - After that, patching is applied in the current application and in all others that use the same environment.
   * - Disable patching notifications
     - Use ``--no-verbose`` or ``-nv`` keys:

       ::

          python -m sklearnex.glob patch_sklearn -a svc random_forest_classifier -nv
     -
   * - Disable global patching
     - Run this command:

       ::

          python -m sklearnex.glob unpatch_sklearn
     -
   * - Disable global patching via code
     - Use the ``global_patch`` argument in the ``unpatch_sklearn`` function

       ::

          from sklearnex import unpatch_sklearn
          unpatch_sklearn(global_patch=True)
     -

.. tip::

    Pass ``verbose=True`` to make it print a message confirming that the |sklearnex| is being used when importing ``sklearn``.

.. Note::

    If you clone an environment with enabled global patching, it will already be applied in the new environment.


API Reference
=============

.. autofunction:: sklearnex.patch_sklearn

.. autofunction:: sklearnex.unpatch_sklearn

.. autofunction:: sklearnex.sklearn_is_patched

.. autofunction:: sklearnex.is_patched_instance

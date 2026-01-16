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

The easiest way to call optimized routines from the |sklearnex| is by patching the
``sklearn`` module from |sklearn|, which can be achieved with just a call to
:obj:`sklearnex.patch_sklearn`.

See also :ref:`patching` for further ways of using the |sklearnex|.

Example usage
=============

To patch all estimators, just import ``patch_sklearn`` and call it:

.. code:: python

    from sklearnex import is_patched_instance
    from sklearnex import patch_sklearn
    from sklearn.linear_model import LinearRegression
    
    is_patched_instance(LinearRegression()) == False

    patch_sklearn()
    from sklearn.linear_model import LinearRegression # now calls sklearnex
    is_patched_instance(LinearRegression()) == True


API Reference
=============

.. autofunction:: sklearnex.patch_sklearn

.. autofunction:: sklearnex.unpatch_sklearn

.. autofunction:: sklearnex.sklearn_is_patched

.. autofunction:: sklearnex.is_patched_instance

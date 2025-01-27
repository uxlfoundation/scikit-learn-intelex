.. ******************************************************************************
.. * Copyright 2021 Intel Corporation
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

.. _global_patching:

###############
Global Patching
###############

Use global patching to patch all your scikit-learn applications without any additional actions.

.. rubric:: Prerequisites

- |intelex|
- Scikit-learn
- read and write permissions to Scikit-learn files

Patch all supported algorithms
===============================

To patch all :ref:`supported algorithms <sklearn_algorithms>`, run::

    python -m sklearnex.glob patch_sklearn

Patch selected algorithms
=========================

If you want to patch only some algorithms, use ``--algorithm`` or ``-a`` keys
with a list of algorithms to patch.

For example, to patch only SVC and RandomForestClassifier estimators, run::

    python -m sklearnex.glob patch_sklearn -a svc random_forest_classifier

Disable patching notifications
==============================

If you do not want to receive patching notifications, then use ``--no-verbose`` or ``-nv`` keys::

    python -m sklearnex.glob patch_sklearn -a svc random_forest_classifier -nv

.. note::
    If you run the global patching command several times with different parameters,
    then only the last configuration will be applied.

Disable global patching
=======================

To disable global patching, use the following command::

    python -m sklearnex.glob unpatch_sklearn

Enable global patching via code
===============================

You can also enable global patching in your code. To do this,
use the ``patch_sklearn`` function with the ``global_patch`` argument::

    from sklearnex import patch_sklearn
    patch_sklearn(global_patch=True)
    import sklearn

After that, Scikit-learn patches will be enabled in the current application and
in all others that use the same environment.

Disable global patching via code
================================

To disable global patching via code, use the ``global_patch``
argument in the ``unpatch_sklearn`` function::

    from sklearnex import unpatch_sklearn
    unpatch_sklearn(global_patch=True)

.. note::
    If you clone an environment with enabled global patching, it will already be applied in the new environment.

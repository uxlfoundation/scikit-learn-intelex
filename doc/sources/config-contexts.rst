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
.. _config_contexts:

=========================================
Configuration Contexts and Global Options
=========================================

Overview
========

Just like |sklearn|, the |sklearnex| offers configurable options which can be managed
locally through a configuration context, or globally through process-wide settings,
by extending the configuration-related functions from |sklearn| (see :obj:`sklearn.config_context`
for details).

Configurations in the |sklearnex| are particularly useful for :ref:`GPU functionalities <oneapi_gpu>`
and :ref:`SMPD mode <distributed>`, and are necessary to modify for enabling :ref:`array API <array_api>`.

Configuration context and global options manager for the |sklearnex| can either be imported directly
from the module ``sklearnex``, or can be imported from the ``sklearn`` module after applying patching.

Note that options in the |sklearnex| are a superset of options from |sklearn|, and options passed to
the configuration contexts and global settings of the |sklearnex| will also affect |sklearn| if the
option is supported by it - meaning: the same context manager  or global option setter is used for
both libraries.

Example usage
=============

Example using the ``target_offload`` option to make computations run on a GPU:

With a local context
--------------------

Here, only the operations from |sklearn| and from the |sklearnex| that happen within the 'with'
block will be affected by the options:

.. code:: python

    import numpy as np
    from sklearnex import config_context
    from sklearnex.cluster import DBSCAN

    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                  [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    with config_context(target_offload="gpu"):
        clustering = DBSCAN(eps=3, min_samples=2).fit(X)

As a global option
------------------

Here, all computations from |sklearn| and from the |sklearnex| that happen after the option
is modified are affected:

.. code:: python

    import numpy as np
    from sklearnex import set_config
    from sklearnex.cluster import DBSCAN

    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                  [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    
    set_config(target_offload="gpu") # set it globally
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    set_config(target_offload="auto") # restore it back

API Reference
=============

Note that all of the options accepted by these functions in |sklearn| are also accepted
here - these just list the additional options offered by the |sklearnex|.

.. autofunction:: sklearnex.config_context

.. autofunction:: sklearnex.get_config

.. autofunction:: sklearnex.set_config

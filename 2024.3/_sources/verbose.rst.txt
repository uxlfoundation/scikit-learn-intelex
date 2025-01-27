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

.. _verbose:

############
Verbose Mode
############

Use |intelex| in verbose mode to find out which implementation of the algorithm is currently used,
|intelex| or original Scikit-learn.

.. note:: Verbose mode is only available for :ref:`supported algorithms <sklearn_algorithms>`.

To enable verbose mode, set the ``SKLEARNEX_VERBOSE`` environment variable as shown below:

- On Linux and MacOS::

     export SKLEARNEX_VERBOSE=INFO

- On Windows::

     set SKLEARNEX_VERBOSE=INFO

Alternatively, get |intelex| logger and set its logging level in the Python code::

     import logging
     logger = logging.getLogger('sklearnex')
     logger.setLevel(logging.INFO)

During the calls that use Intel-optimized scikit-learn, you will receive additional print statements
that indicate which implementation is being called.
These print statements are only available for :ref:`supported algorithms <sklearn_algorithms>`.

For example, for DBSCAN you get one of these print statements depending on which implementation is used::

    INFO:sklearnex: sklearn.cluster.DBSCAN.fit: running accelerated version on CPU

::

    INFO:sklearnex: sklearn.cluster.DBSCAN.fit: fallback to original Scikit-learn

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

=========================
Installation Instructions
=========================

The |sklearnex| can be easily installed as a Python package under name ``scikit-learn-intelex`` from the most
common distribution channels for Python. For GPU support, an additional package ``scikit-learn-intelex-gpu`` is
also made available through the same channels - see :doc:`oneapi-gpu` for more information.

Binary wheels and conda packages are provided for the x86-64 architecture, for both Windows* and Linux*, for the
Python versions supported by current versions of |sklearn| - see :doc:`about` for more information.

Installing from PyPI
--------------------

To install with ``pip`` from the Python Package Index (PyPI), execute the following command on a terminal: ::

  pip install scikit-learn-intelex

.. tip:: To prevent version conflicts, we recommend creating and activating a new virtual environment for |sklearnex|.

.. tip:: Wheels are also available through Intel's index: ``https://software.repos.intel.com/python/pypi``

Installing from conda-forge
---------------------------

To install the |sklearnex| in a fresh conda environment, execute the following command in a terminal: ::

  conda create -n sklex -c conda-forge --override-channels scikit-learn-intelex
  conda activate sklex

To install in an existing environment: ::

  conda install -c conda-forge --override-channels scikit-learn-intelex

.. hint::
  It is advisable to use the `miniforge <https://github.com/conda-forge/miniforge>`__ installer for ``conda``/``mamba``, as it comes with
  ``conda-forge`` as the default channel.

.. warning::
  The main Anaconda channel also provides distributions of ``scikit-learn-intelex``, but it does not provide the latest versions, nor does
  it provide GPU-enabled builds. It is highly recommended to install ``scikit-learn-intelex`` from either conda-forge or from Intel's channel instead.

.. tip::

  The |sklearnex| is also available at Intel's conda channel: ``https://software.repos.intel.com/python/conda``

  Packages from the Intel channel, which include also optimized versions of NumPy and SciPy, are meant to be used together with
  dependencies from the **conda-forge** channel, and might not work correctly when used in an environment where packages from the
  ``anaconda`` default channel have been installed.


Building from Sources
---------------------

The |sklearnex| is a fully open-source package (`link to source code <https://github.com/uxlfoundation/scikit-learn-intelex>`_) which
can be compiled from source with standard compiler toolkits.

See :doc:`building-from-source` for details.

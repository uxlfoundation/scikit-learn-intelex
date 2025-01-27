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

.. |intelex_repo| replace:: |intelex| repository
.. _intelex_repo: https://github.com/intel/scikit-learn-intelex

#############
Installation
#############

|intelex| is available at the `Python Package Index <https://pypi.org/project/scikit-learn-intelex/>`_,
on Anaconda Cloud in `Conda-Forge channel <https://anaconda.org/conda-forge/scikit-learn-intelex>`_ and
in `Intel channel <https://anaconda.org/intel/scikit-learn-intelex>`_.

|intelex| is also available as a part of `Intel AI Analytics Toolkit
<https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit.html>`_ (AI Kit).
If you already have AI Kit installed, you do not need to separately install the extension.

You can also build the extension from :ref:`sources <build_from_sources>`.

.. seealso:: Check out :ref:`system_requirements` before you start the installation process.

Install from distribution channels
-----------------------------------

.. contents:: :local:

Install from PyPI (recommended by default)
===========================================

#. **[Optional step] [Recommended]** To prevent version conflicts, create and activate a new environment:

   .. tabs::

      .. tab:: Linux

         ::

            python -m venv env
            source env/bin/activate

      .. tab:: Windows

         ::

            python -m venv env
            .\env\Scripts\activate

#. Install `scikit-learn-intelex`:

   ::

      pip install scikit-learn-intelex

Install from Anaconda Cloud
============================

To prevent version conflicts, we recommend installing `scikit-learn-intelex` into a new conda environment.
For each distribution channel, there are two installation options: with and without a creation of a new environment.

- Install from **Anaconda Cloud: Conda-Forge channel** (recommended by default for conda users):

  - into a newly created environment (recommended)::

       conda create -n env -c conda-forge python=3.9 scikit-learn-intelex

    .. important::

       If you do not specify the version of Python (``python=3.9`` in the example above), then Python 3.10 is downloaded by default,
       which is not supported.

       See :ref:`supported configurations for Conda-Forge channel <sys_req_conda_forge>`.

  - into your current environment::

       conda install scikit-learn-intelex -c conda-forge

- Install from **Anaconda Cloud: Intel channel** (recommended for the users of Intel® Distribution for Python):

  - into a newly created environment (recommended)::

       conda create -n env -c intel python scikit-learn-intelex

    Note that you may also specify the version of Python to download
    (see :ref:`supported configurations for Anaconda Intel channel <sys_req_conda_intel>`).
    For example::

       conda create -n env -c intel python=3.7 scikit-learn-intelex

  - into your current environment::

       conda install scikit-learn-intelex -c intel

- Install from **Anaconda Cloud: Main channel**:

  - into a newly created environment (recommended)::

       conda create -n env python=3.9 scikit-learn-intelex

    .. important::

       If you do not specify the version of Python (``python=3.9`` in the example above), then Python 3.10 is downloaded by default,
       which is not supported.

       See :ref:`supported configurations for Anaconda main channel <sys_req_conda_main>`.

  - into your current environment::

       conda install scikit-learn-intelex

.. _build_from_sources:

Build from Sources
---------------------

Sources are available in |intelex_repo|_.

.. rubric:: Prerequisites

::

    Python version >= 3.6, < 3.10
    daal4py >= 2021.4

.. note::
    You can `build daal4py from sources <https://github.com/intel/scikit-learn-intelex/blob/main/daal4py/INSTALL.md>`_ or get it from `distribution channels
    <https://intelpython.github.io/daal4py/#getting-daal4py>`_.

Configure the build with environment variables
==============================================

::

    SKLEARNEX_VERSION: sets package version

Build |intelex|
===============

To install the package::

    cd <checkout-dir>
    python setup_sklearnex.py install

To install the package in the development mode::

    cd <checkout-dir>
    python setup.py develop

To install scikit-learn-intelex without downloading daal4py::

    cd <checkout-dir>
    python setup_sklearnex.py install --single-version-externally-managed --record=record.txt

To install scikit-learn-intelex without downloading daal4py in the development mode::

    cd <checkout-dir>
    python setup_sklearnex.py develop --no-deps

.. note::
    The ``develop`` mode will not install the package but it will create a ``.egg-link`` in the deployment directory
    back to the project source code directory. That way you can edit the source code and see the changes
    without having to reinstall the package every time you make a small change.

⚠️ Keys ``--single-version-externally-managed`` and ``--no-deps`` are required so that daal4py is not downloaded after installation of |intelex|

.. note::
    ``--single-version-externally-managed`` is an option used for Python packages instructing the setuptools module
    to create a Python package that can be easily managed by the package manager on the host

Next Steps
==========

- :ref:`What is patching and how to patch scikit-learn <what_is_patching>`

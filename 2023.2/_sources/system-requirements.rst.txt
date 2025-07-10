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

.. _system_requirements:

#################################################
System requirements and supported configurations
#################################################

This page provides details about hardware, operating system, and software prerequisites for |intelex|.

Supported configurations
------------------------

|intelex| supports optimizations for the last four versions of scikit-learn.
The latest release of scikit-learn-intelex-2023.1.X supports scikit-learn 0.24.X, 1.0.X, 1.1.X and 1.2.X.

|intelex| is available for installation from different channels. 
There is a difference in supported configurations for each distribution channel.

.. _sys_req_pip:

PyPI channel
=============

.. list-table::
   :widths: 25 8 8 8 8 8
   :header-rows: 1
   :align: left

   * - OS / Python version
     - Python 3.7
     - Python 3.8
     - Python 3.9
     - Python 3.10
     - Python 3.11
   * - Linux
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
   * - Windows
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
   * - macOS
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]

.. _sys_req_conda_forge:

Anaconda Cloud: Conda-Forge channel
===================================

.. list-table::
   :widths: 25 8 8 8 8
   :header-rows: 1
   :align: left

   * - OS / Python version
     - Python 3.8
     - Python 3.9
     - Python 3.10
     - Python 3.11
   * - Linux
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]
   * - Windows
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]
   * - macOS
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]

.. _sys_req_conda_intel:

Anaconda Cloud: Intel channel
==============================

.. list-table::
   :widths: 25 8 8 8 8 8
   :header-rows: 1
   :align: left

   * - OS / Python version
     - Python 3.7
     - Python 3.8
     - Python 3.9
     - Python 3.10
     - Python 3.11
   * - Linux
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
   * - Windows
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
     - [CPU, GPU]
   * - macOS
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]

.. _sys_req_conda_main:

Anaconda Cloud: Main channel
==============================

.. list-table::
   :widths: 25 8 8 8 8
   :header-rows: 1
   :align: left

   * - OS / Python version
     - Python 3.8
     - Python 3.9
     - Python 3.10
     - Python 3.11
   * - Linux
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]
   * - Windows
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]
   * - macOS
     - [CPU]
     - [CPU]
     - [CPU]
     - [CPU]

System Requirements
-------------------

For CPU
=======

.. rubric:: Operating systems

The last two versions of the following operating systems:

- **Linux**\*
- **Windows**\* and **Windows\* Server**
- **macOS**\*

.. rubric:: Hardware platforms

- All processors with x86 architecture

.. note::
    Your processor must support at least one of SSE2, SSE4.2, AVX2, AVX512 instruction sets.

.. note::
    ARM* architecture is not supported.

.. note::
    Intel® processors provide better performance then other CPUs.
    Read more about hardware comparison in our :ref:`blogs <blogs>`.

For GPU
=======

.. important::
    If you are using accelerators, please refer to the DPC++ compiler system
    requirements `here <https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html>`_

.. rubric:: Operating systems

The last two versions of the following operating systems:

- **Linux**\*
- **Windows**\* and **Windows\* Server**

.. rubric:: Hardware platforms

- All Intel® integrated and discrete GPUs
- `Intel® GPU drivers <https://www.intel.com/content/www/us/en/develop/articles/installation-guide-for-intel-oneapi-toolkits.html#installGPUdriver>`__

.. seealso:: :ref:`oneapi_gpu`

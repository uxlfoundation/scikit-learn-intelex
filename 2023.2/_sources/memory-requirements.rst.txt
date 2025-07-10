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

.. _memory_requirements:

###################
Memory Requirements 
###################

By default, algorithms in |intelex| run in multi-thread mode, which uses all available threads. 
This may lead to optimized algorithms consuming more RAM than the corresponding stock scikit-learn algorithms.

.. list-table::
   :widths: 10 30 30
   :header-rows: 1
   :align: left

   * - Algorithm
     - Single-thread mode
     - Multi-thread mode
   * - SVM
     - Both Scikit-learn and |intelex| consume approximately the same amount of RAM.
     - In |intelex|, an algorithm with ``N`` threads will consume ``N`` times more RAM.

Memory consumption on GPU
-------------------------
In all |intelex| algorithms with GPU support, computations run on device memory. 
The device memory must be large enough to store a copy of the entire dataset.
Additional device memory may also be required for internal arrays used in computation.

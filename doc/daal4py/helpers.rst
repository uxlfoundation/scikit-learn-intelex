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

Thread control and MPI helpers
==============================

Thread control
--------------

Documentation for functions that control the global thread settings in ``daal4py``:

.. autofunction:: daal4py.daalinit
.. autofunction:: daal4py.num_threads
.. autofunction:: daal4py.enable_thread_pinning

MPI helpers
-----------

Documentation for helper functions that can be used in distributed mode, particularly when using MPI without ``mpi4py``:

.. autofunction:: daal4py.daalfini
.. autofunction:: daal4py.num_procs
.. autofunction:: daal4py.my_procid

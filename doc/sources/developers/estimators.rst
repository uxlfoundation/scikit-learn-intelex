.. ******************************************************************************
.. * Copyright 2024 Intel Corporation
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
.. _estimators:

==========
Estimators
==========
TBD

daal4py
=======
TBD

onedal4py
=========

`to_table`
----------
In data processing stage `to_table` can be used as a step to transform input data to backend
oneDAL lib table format before feeding it into the backend functions, ensuring compatibility
and consistency.

The input data could be a NumPy ndarray, DPCTL usm_ndarray, DPNP ndarray or a CSR matrix or array.
You want to ensure that all data is converted to a table format before passing them backend functions.

If using the sua_iface parameter, ensure that the backend is correctly set up and compatible
with the data being processed. Different backends may have specific requirements or optimizations.

TBD

.. code-block:: python

   # Using to_table to convert different formats to oneDAL table

   data_array = np.array([[7, 8, 9], [10, 11, 12]])
   table_from_array = to_table(data_array)

   # Example CSR matrix
   row = np.array([0, 0, 1, 2, 2, 2])
   col = np.array([0, 2, 2, 0, 1, 2])
   data = np.array([1, 2, 3, 4, 5, 6])
   data_csr = csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
   table_from_csr = to_table(data_csr)

sklearnex
=========
TBD
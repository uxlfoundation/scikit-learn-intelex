.. Copyright 2020 Intel Corporation
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

.. _data:

##########
Input Data
##########

.. include:: note.rst

All array arguments to compute functions and to algorithm constructors can be
provided in different formats. daal4py will automatically do its best to work on
the provided data with minimal overhead, most notably without copying the data.

Numpy Arrays
------------
daal4py can directly handle all types of numpy arrays with numerical data
without copying the entire data. Arrays can be homogeneous (e.g. simple dtype) or
heterogeneous (structured array) as well as contiguous or non-contiguous.

Pandas DataFrames
-----------------
daal4py directly accepts pandas DataFrames with columns of numerical data. No
extra full copy is required.

SciPy Sparse CSR Matrix
-----------------------
daal4py can directly handle matrices of type scipy.sparse.csr_matrix without
copying the entire data.

Note: some algorithms can be configured to use an optimized compute path for CSR
data. It is required to explicitly specify the CSR method, otherwise the default
and less efficient method is used.

CSV Files
---------
The compute functions daal4py's algorithms additionally accept
CSV-filenames. Internally, daal4py will use DAAL's fast CSV reader to create
contiguous homogeneous tables.

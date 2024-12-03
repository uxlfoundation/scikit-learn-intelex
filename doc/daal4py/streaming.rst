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

.. _streaming:

##############
Streaming Data
##############

.. include:: note.rst
  
For large quantities of data it might be impossible to provide all input data at
once. This might be because the data resides in multiple files and merging it is
to costly (or not feasible in other ways). In other cases the data is simply too
large to be loaded completely into memory. Or, the data might come in as an
actual stream. daal4py's streaming mode allows you to process such data.

Besides supporting certain use cases, streaming also allows interleaving I/O
operations with computation.

daal4py's streaming mode is as easy as follows:

1. When constructing the algorithm configure it with ``streaming=True``::

     algo = daal4py.svd(streaming=True)
2. Repeat calling ``compute(input-data)`` with chunks of your input (arrays, DataFrames or
   files)::

     for f in input_files:
         algo.compute(f)
3. When done with inputting, call ``finalize()`` to obtain the result::

     result = algo.finalize()

The streaming algorithms also accept arrays and DataFrames as input, e.g. the
data can come from a stream rather than from multiple files. Here is an example
which simulates a data stream using a generator which reads a file in chunks:
`SVD reading stream of data <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/stream.py>`_

Supported Algorithms and Examples
---------------------------------
The following algorithms support streaming:

- SVD (svd)

  - `SVD <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/svd_streaming.py>`_

- Linear Regression Training (linear_regression_training)

  - `Linear Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/linear_regression_streaming.py>`_

- Ridge Regression Training (ridge_regression_training)

  - `Ridge Regression <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/ridge_regression_streaming.py>`_

- Multinomial Naive Bayes Training (multinomial_naive_bayes_training)

  - `Naive Bayes <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/naive_bayes_streaming.py>`_

- Moments of Low Order

  - `Low Order Moments <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/low_order_moms_streaming.py>`_

- Covariance

  - `Covariance <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/covariance_streaming.py>`_

- QR

  - `QR <https://github.com/uxlfoundation/scikit-learn-intelex/tree/main/examples/daal4py/qr_streaming.py>`_

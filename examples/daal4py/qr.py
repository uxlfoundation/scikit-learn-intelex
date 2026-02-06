# ==============================================================================
# Copyright 2014 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# daal4py QR example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "qr.csv"

    # configure a QR object
    algo = d4p.qr()

    # let's provide a file directly, not a table/array
    result1 = algo.compute(str(infile))

    # We can also load the data ourselves and provide the numpy array
    data = readcsv(infile)
    result2 = algo.compute(data)

    # QR result provide matrixQ and matrixR
    assert result1.matrixQ.shape == data.shape
    assert result1.matrixR.shape == (data.shape[1], data.shape[1])

    assert np.allclose(result1.matrixQ, result2.matrixQ, atol=1e-07)
    assert np.allclose(result1.matrixR, result2.matrixR, atol=1e-07)

    if hasattr(data, "toarray"):
        data = data.toarray()  # to make the next assertion work with scipy's csr_matrix
    assert np.allclose(data, np.matmul(result1.matrixQ, result1.matrixR))

    return data, result1


if __name__ == "__main__":
    _, result = main()
    print(result)
    print("All looks good!")

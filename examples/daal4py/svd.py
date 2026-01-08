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

# daal4py SVD example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "svd.csv"

    # configure a SVD object
    algo = d4p.svd()

    # let's provide a file directly, not a table/array
    result1 = algo.compute(str(infile))

    # We can also load the data ourselves and provide the numpy array
    algo = d4p.svd()
    data = readcsv(infile, usecols=range(18), dtype=np.float32)
    result2 = algo.compute(data)

    # SVD result objects provide leftSingularMatrix,
    # rightSingularMatrix and singularValues
    assert np.allclose(result1.leftSingularMatrix, result2.leftSingularMatrix, atol=1e-07)
    assert np.allclose(
        result1.rightSingularMatrix, result2.rightSingularMatrix, atol=1e-07
    )
    assert np.allclose(result1.singularValues, result2.singularValues, atol=1e-07)
    assert result1.singularValues.shape == (1, data.shape[1])
    assert result1.rightSingularMatrix.shape == (data.shape[1], data.shape[1])
    assert result1.leftSingularMatrix.shape == data.shape

    if hasattr(data, "toarray"):
        data = data.toarray()  # to make the next assertion work with scipy's csr_matrix
    assert np.allclose(
        data,
        np.matmul(
            np.matmul(result1.leftSingularMatrix, np.diag(result1.singularValues[0])),
            result1.rightSingularMatrix,
        ),
    )

    return (data, result1)


if __name__ == "__main__":
    (_, result) = main()
    print(result)
    print("All looks good!")

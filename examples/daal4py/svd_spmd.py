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

# daal4py SVD example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./svd_spmd.py

from pathlib import Path

from numpy import allclose, loadtxt

import daal4py as d4p


def main():
    # Each process gets its own data
    data_path = Path(__file__).parent / "data" / "distributed"
    infile = data_path / f"svd_{d4p.my_procid()+ 1}.csv"

    # configure a SVD object
    algo = d4p.svd(distributed=True)

    # let's provide a file directly, not a table/array
    result1 = algo.compute(str(infile))

    # We can also load the data ourselves and provide the numpy array
    data = loadtxt(infile, delimiter=",")
    result2 = algo.compute(data)

    # SVD result objects provide leftSingularMatrix,
    # rightSingularMatrix and singularValues
    assert result1.leftSingularMatrix.shape == data.shape
    assert result1.singularValues.shape == (1, data.shape[1])
    assert result1.rightSingularMatrix.shape == (data.shape[1], data.shape[1])

    assert allclose(result1.leftSingularMatrix, result2.leftSingularMatrix, atol=1e-05)
    assert allclose(result1.rightSingularMatrix, result2.rightSingularMatrix, atol=1e-05)
    assert allclose(result1.singularValues, result2.singularValues, atol=1e-05)

    return data, result1


if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()
    (_, result) = main()
    # result is available on all processes - but we print only on root
    if d4p.my_procid() == 0:
        print(
            "\nEach process has singularValues and rightSingularMatrix "
            "but only his part of leftSingularMatrix:\n"
        )
        print(result)
        print("All looks good!")
    d4p.daalfini()

# ===============================================================================
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
# ===============================================================================

# daal4py LBFGS (limited memory Broyden-Fletcher-Goldfarb-Shanno)
# example for shared memory systems
# using cross entropy loss function

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    nFeatures = 6
    nClasses = 5
    nIterations = 1000
    stepLength = 1.0e-4

    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "lbfgs_cr_enr_loss.csv"

    # Read the data
    data = readcsv(infile, range(nFeatures))
    dep_data = readcsv(infile, range(nFeatures, nFeatures + 1))
    nVectors = data.shape[0]

    # configure a function
    func = d4p.optimization_solver_cross_entropy_loss(
        nClasses, nVectors, interceptFlag=True
    )
    func.setup(data, dep_data)

    # configure a algorithm
    stepLengthSequence = np.array([[stepLength]], dtype=np.double)
    alg = d4p.optimization_solver_lbfgs(
        func, stepLengthSequence=stepLengthSequence, nIterations=nIterations
    )

    # do the computation
    nParameters = nClasses * (nFeatures + 1)
    initialPoint = np.full((nParameters, 1), 0.001, dtype=np.double)
    res = alg.compute(initialPoint)

    # result provides minimum and nIterations
    assert res.minimum.shape == (nParameters, 1)
    assert res.nIterations[0][0] <= nIterations

    return res


if __name__ == "__main__":
    res = main()
    print(
        "\nExpected coefficients:\n",
        np.array(
            [
                [-2.277],
                [2.836],
                [14.985],
                [0.511],
                [7.510],
                [-2.831],
                [-5.814],
                [-0.033],
                [13.227],
                [-24.447],
                [3.730],
                [10.394],
                [-10.461],
                [-0.766],
                [0.077],
                [1.558],
                [-1.133],
                [2.884],
                [-3.825],
                [7.699],
                [2.421],
                [-0.135],
                [-6.996],
                [1.785],
                [-2.294],
                [-9.819],
                [1.692],
                [-0.725],
                [0.069],
                [-8.41],
                [1.458],
                [-3.306],
                [-4.719],
                [5.507],
                [-1.642],
            ],
            dtype=np.double,
        ),
    )
    print("\nResulting coefficients:\n", res.minimum)
    print("\nNumber of iterations performed:\n", res.nIterations[0][0])
    print("All looks good!")

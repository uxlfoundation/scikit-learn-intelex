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

# daal4py low order moments example for shared memory systems

from pathlib import Path

import numpy as np

import daal4py as d4p
from daal4py.sklearn.utils import pd_read_csv


def main(readcsv=pd_read_csv, method="defaultDense"):
    # read data from file
    data_path = Path(__file__).parent / "data" / "batch"
    file = data_path / "covcormoments_dense.csv"
    data = readcsv(file, range(10))

    # compute
    alg = d4p.low_order_moments(method=method)
    res = alg.compute(data)

    # result provides minimum, maximum, sum, sumSquares, sumSquaresCentered,
    # mean, secondOrderRawMoment, variance, standardDeviation, variation
    assert all(
        getattr(res, name).shape == (1, data.shape[1])
        for name in [
            "minimum",
            "maximum",
            "sum",
            "sumSquares",
            "sumSquaresCentered",
            "mean",
            "secondOrderRawMoment",
            "variance",
            "standardDeviation",
            "variation",
        ]
    )

    return res


if __name__ == "__main__":
    res = main()
    # print results
    print("\nMinimum:\n", res.minimum)
    print("\nMaximum:\n", res.maximum)
    print("\nSum:\n", res.sum)
    print("\nSum of squares:\n", res.sumSquares)
    print("\nSum of squared difference from the means:\n", res.sumSquaresCentered)
    print("\nMean:\n", res.mean)
    print("\nSecond order raw moment:\n", res.secondOrderRawMoment)
    print("\nVariance:\n", res.variance)
    print("\nStandard deviation:\n", res.standardDeviation)
    print("\nVariation:\n", res.variation)
    print("All looks good!")

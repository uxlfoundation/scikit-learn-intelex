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

# daal4py Stump regression example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "stump_train.csv"
    testfile = data_path / "stump_test.csv"

    # Configure a stump regression training object
    train_algo = d4p.stump_regression_training()

    # Read data. Let's have 20 independent,
    # and 1 dependent variable (for each observation)
    indep_data = readcsv(infile, range(20))
    dep_data = readcsv(infile, range(20, 21))
    # Now train/compute, the result provides the model for prediction
    train_result = train_algo.compute(indep_data, dep_data)

    # Now let's do some prediction
    predict_algo = d4p.stump_regression_prediction()
    # read test data (with same #features)
    pdata = readcsv(testfile, range(20))
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # The prediction result provides prediction
    assert predict_result.prediction.shape == (pdata.shape[0], dep_data.shape[1])
    ptdata = np.loadtxt(testfile, usecols=range(20, 21), delimiter=",", ndmin=2)
    assert np.allclose(predict_result.prediction, ptdata)

    return (train_result, predict_result, ptdata)


if __name__ == "__main__":
    train_result, predict_result, ptdata = main()
    print("\nGround truth (first 20 observations):\n", ptdata[:20])
    print(
        "Stump regression results: (first 20 observations):\n",
        predict_result.prediction[:20],
    )
    print("All looks good!")

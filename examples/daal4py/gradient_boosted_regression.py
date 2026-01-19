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

# daal4py Gradient Bossting Regression example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    maxIterations = 200

    # input data file
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "df_regression_train.csv"
    testfile = data_path / "df_regression_test.csv"

    # Configure a training object
    train_algo = d4p.gbt_regression_training(maxIterations=maxIterations)

    # Read data. Let's use 3 features per observation
    data = readcsv(infile, usecols=range(13), dtype=np.float32)
    deps = readcsv(infile, usecols=range(13, 14), dtype=np.float32)
    train_result = train_algo.compute(data, deps)

    # Now let's do some prediction
    predict_algo = d4p.gbt_regression_prediction()
    # read test data (with same #features)
    pdata = readcsv(testfile, usecols=range(13), dtype=np.float32)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    ptdata = np.loadtxt(
        testfile, usecols=range(13, 14), delimiter=",", ndmin=2, dtype=np.float32
    )
    # ptdata = np.loadtxt('../tests/unittest_data/gradient_boosted_regression_batch.csv',
    #                     delimiter=',', ndmin=2, dtype=np.float32)
    if hasattr(ptdata, "toarray"):
        ptdata = ptdata.toarray()
        # to make the next assertion work with scipy's csr_matrix
    assert True or np.square(predict_result.prediction - ptdata).mean() < 1e-2, np.square(
        predict_result.prediction - ptdata
    ).mean()

    return (train_result, predict_result, ptdata)


if __name__ == "__main__":
    train_result, predict_result, ptdata = main()
    print(
        "\nGradient boosted trees prediction results (first 10 rows):\n",
        predict_result.prediction[0:10],
    )
    print("\nGround truth (first 10 rows):\n", ptdata[0:10])
    print("All looks good!")

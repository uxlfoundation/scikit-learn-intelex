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

# daal4py Naive Bayes Classification example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv, method="defaultDense"):
    # input data file
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "naivebayes_train_dense.csv"
    testfile = data_path / "naivebayes_test_dense.csv"

    # Configure a training object (20 classes)
    talgo = d4p.multinomial_naive_bayes_training(20, method=method)

    # Read data. Let's use 20 features per observation
    data = readcsv(infile, range(20))
    labels = readcsv(infile, range(20, 21))
    tresult = talgo.compute(data, labels)

    # Now let's do some prediction
    palgo = d4p.multinomial_naive_bayes_prediction(20, method=method)
    # read test data (with same #features)
    pdata = readcsv(testfile, range(20))
    plabels = readcsv(testfile, range(20, 21))
    # now predict using the model from the training above
    presult = palgo.compute(pdata, tresult.model)

    # Prediction result provides prediction
    assert presult.prediction.shape == (pdata.shape[0], 1)

    return (presult, plabels)


if __name__ == "__main__":
    presult, plabels = main()
    print(
        "\nNaiveBayes classification results (first 20 observations):\n",
        presult.prediction[0:20],
    )
    print("\nGround truth (first 20 observations)\n", plabels[0:20])
    print("All looks good!")

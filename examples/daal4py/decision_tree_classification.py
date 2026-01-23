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

# daal4py Decision Tree Classification example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    # input data file
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "decision_tree_train.csv"
    prunefile = data_path / "decision_tree_prune.csv"
    testfile = data_path / "decision_tree_test.csv"

    # Configure a training object (5 classes)
    train_algo = d4p.decision_tree_classification_training(5)

    # Read data. Let's use 5 features per observation
    data = readcsv(infile, usecols=range(5), dtype=np.float32)
    labels = readcsv(infile, usecols=range(5, 6), dtype=np.float32)
    prunedata = readcsv(prunefile, usecols=range(5), dtype=np.float32)
    prunelabels = readcsv(prunefile, usecols=range(5, 6), dtype=np.float32)
    train_result = train_algo.compute(data, labels, prunedata, prunelabels)

    # Now let's do some prediction
    predict_algo = d4p.decision_tree_classification_prediction()
    # read test data (with same #features)
    pdata = readcsv(testfile, usecols=range(5), dtype=np.float32)
    plabels = readcsv(testfile, usecols=range(5, 6), dtype=np.float32)
    # now predict using the model from the training above
    predict_result = predict_algo.compute(pdata, train_result.model)

    # Prediction result provides prediction
    assert predict_result.prediction.shape == (pdata.shape[0], 1)

    return (train_result, predict_result, plabels)


if __name__ == "__main__":
    train_result, predict_result, plabels = main()
    print(
        "\nDecision tree prediction results (first 20 rows):\n",
        predict_result.prediction[0:20],
    )
    print("\nGround truth (first 20 rows):\n", plabels[0:20])
    print("All looks good!")

# ===============================================================================
# Copyright 2020 Intel Corporation
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

# daal4py Brute Force KNN example for shared memory systems

import os
from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    # Input data set parameters
    data_path = Path(__file__).parent / "data" / "batch"
    train_file = data_path / "k_nearest_neighbors_train.csv"
    predict_file = data_path / "k_nearest_neighbors_test.csv"

    # Read data. Let's use 5 features per observation
    nFeatures = 5
    nClasses = 5
    train_data = readcsv(train_file, range(nFeatures))
    train_labels = readcsv(train_file, range(nFeatures, nFeatures + 1))

    # Create an algorithm object and call compute
    train_algo = d4p.bf_knn_classification_training(nClasses=nClasses)
    # 'weights' is optional argument, let's use equal weights
    # in this case results must be the same as without weights
    weights = np.ones((train_data.shape[0], 1))
    train_result = train_algo.compute(train_data, train_labels, weights)

    # Now let's do some prediction
    predict_data = readcsv(predict_file, range(nFeatures))
    predict_labels = readcsv(predict_file, range(nFeatures, nFeatures + 1))

    # Create an algorithm object and call compute
    predict_algo = d4p.bf_knn_classification_prediction(nClasses=nClasses)
    predict_result = predict_algo.compute(predict_data, train_result.model)

    # We expect less than 170 mispredicted values
    assert np.count_nonzero(predict_labels != predict_result.prediction) < 170

    return (train_result, predict_result, predict_labels)


if __name__ == "__main__":
    train_result, predict_result, predict_labels = main()
    print("Brute Force kNN classification results:")
    print("Ground truth(observations #30-34):\n", predict_labels[30:35])
    print(
        "Classification results(observations #30-34):\n", predict_result.prediction[30:35]
    )

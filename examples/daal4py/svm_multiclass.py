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

# daal4py multi-class SVM example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    nFeatures = 20
    nClasses = 5

    # read training data from file
    # with nFeatures features per observation and 1 class label
    data_path = Path(__file__).parent / "data" / "batch"
    train_file = data_path / "svm_multi_class_train_dense.csv"
    train_data = readcsv(train_file, range(nFeatures))
    train_labels = readcsv(train_file, range(nFeatures, nFeatures + 1))
    lin_kernel = d4p.kernel_function_linear()

    # Create and configure algorithm object
    algorithm = d4p.multi_class_classifier_training(
        nClasses=nClasses,
        training=d4p.svm_training(method="thunder", kernel=lin_kernel),
        prediction=d4p.svm_prediction(),
    )

    # Pass data to training. Training result provides model
    train_result = algorithm.compute(train_data, train_labels)
    assert train_result.model.NumberOfFeatures == nFeatures
    assert isinstance(train_result.model.TwoClassClassifierModel(0), d4p.svm_model)

    # Now the prediction stage
    # Read data
    pred_file = data_path / "svm_multi_class_test_dense.csv"
    pred_data = readcsv(pred_file, range(nFeatures))
    pred_labels = readcsv(pred_file, range(nFeatures, nFeatures + 1))

    # Create an algorithm object to predict multi-class SVM values
    algorithm = d4p.multi_class_classifier_prediction(
        nClasses,
        training=d4p.svm_training(method="thunder", kernel=lin_kernel),
        prediction=d4p.svm_prediction(),
    )
    # Pass data to prediction. Prediction result provides prediction
    pred_result = algorithm.compute(pred_data, train_result.model)
    assert pred_result.prediction.shape == (train_data.shape[0], 1)

    return (pred_result, pred_labels)


if __name__ == "__main__":
    (pred_res, pred_labels) = main()
    print(
        "\nSVM classification results (first 20 observations):\n",
        pred_res.prediction[0:20],
    )
    print("\nGround truth (first 20 observations):\n", pred_labels[0:20])
    print("All looks good!")

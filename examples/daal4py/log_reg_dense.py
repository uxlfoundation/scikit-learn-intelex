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

# daal4py logistic regression example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    nClasses = 5
    nFeatures = 6

    # read training data from file with 6 features per observation and 1 class label
    data_path = Path(__file__).parent / "data" / "batch"
    trainfile = data_path / "logreg_train.csv"
    train_data = readcsv(trainfile, range(nFeatures))
    train_labels = readcsv(trainfile, range(nFeatures, nFeatures + 1))

    # set parameters and train
    train_alg = d4p.logistic_regression_training(
        nClasses=nClasses, penaltyL1=0.1, penaltyL2=0.1, interceptFlag=True
    )
    train_result = train_alg.compute(train_data, train_labels)

    # read testing data from file with 6 features per observation
    testfile = data_path / "logreg_test.csv"
    predict_data = readcsv(testfile, range(nFeatures))

    # set parameters and compute predictions
    predict_alg = d4p.logistic_regression_prediction(
        nClasses=nClasses,
        resultsToEvaluate="computeClassLabels|computeClassProbabilities|"
        "computeClassLogProbabilities",
    )
    predict_result = predict_alg.compute(predict_data, train_result.model)
    # the prediction result provides prediction, probabilities and logProbabilities
    assert predict_result.probabilities.shape == (predict_data.shape[0], nClasses)
    assert predict_result.logProbabilities.shape == (predict_data.shape[0], nClasses)
    predict_labels = np.loadtxt(
        testfile, usecols=range(nFeatures, nFeatures + 1), delimiter=",", ndmin=2
    )
    # TODO: Fix multi-class Logistic Regression in oneDAL to get better accuracy
    # and enable this check
    # assert (
    #     np.count_nonzero(predict_result.prediction - predict_labels)
    #     / predict_labels.shape[0]
    #     < 0.025
    # )

    return (train_result, predict_result, predict_labels)


if __name__ == "__main__":
    train_result, predict_result, predict_labels = main()
    print("\nLogistic Regression coefficients:\n", train_result.model.Beta)
    print(
        "\nLogistic regression prediction results (first 10 rows):\n",
        predict_result.prediction[0:10],
    )
    print("\nGround truth (first 10 rows):\n", predict_labels[0:10])
    print(
        "\nLogistic regression prediction probabilities (first 10 rows):\n",
        predict_result.probabilities[0:10],
    )
    print(
        "\nLogistic regression prediction log probabilities (first 10 rows):\n",
        predict_result.logProbabilities[0:10],
    )
    print("All looks good!")

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

# daal4py Naive Bayes Classification example for streaming on shared memory systems

from pathlib import Path

from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv, method="defaultDense"):
    # input data file
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "naivebayes_train_dense.csv"
    testfile = data_path / "naivebayes_test_dense.csv"

    # Configure a training object (20 classes)
    train_algo = d4p.multinomial_naive_bayes_training(20, streaming=True, method=method)

    chunk_size = 250
    lines_read = 0
    # read and feed chunk by chunk
    while True:
        # Read data in chunks
        # Read data. Let's use 20 features per observation
        try:
            data = readcsv(
                infile, usecols=range(20), skip_header=lines_read, max_rows=chunk_size
            )
            labels = readcsv(
                infile, usecols=range(20, 21), skip_header=lines_read, max_rows=chunk_size
            )
        except StopIteration as e:
            if lines_read > 0:
                break
            else:
                raise ValueError("No training data was read - empty input file?") from e
        # Now feed chunk
        train_algo.compute(data, labels)
        lines_read += data.shape[0]

    # All chunks are done, now finalize the computation
    train_result = train_algo.finalize()

    # Now let's do some prediction
    pred_algo = d4p.multinomial_naive_bayes_prediction(20, method=method)
    # read test data (with same #features)
    pred_data = readcsv(testfile, range(20))
    pred_labels = readcsv(testfile, range(20, 21))
    # now predict using the model from the training above
    pred_result = pred_algo.compute(pred_data, train_result.model)

    # Prediction result provides prediction
    assert pred_result.prediction.shape == (pred_data.shape[0], 1)

    return (pred_result, pred_labels)


if __name__ == "__main__":
    result, labels = main()
    print(
        "\nNaiveBayes classification results (first 20 observations):\n",
        result.prediction[0:20],
    )
    print("\nGround truth (first 20 observations)\n", labels[0:20])
    print("All looks good!")

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

# daal4py Naive Bayes Classification example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./naive_bayes_spmd.py

from pathlib import Path

from numpy import loadtxt

import daal4py as d4p


def main():
    # Each process gets its own data
    data_path = Path(__file__).parent / "data" / "distributed"
    infile = data_path / "naivebayes_train_dense.csv"

    # Configure a training object (20 classes)
    talgo = d4p.multinomial_naive_bayes_training(20, distributed=True)

    # Read data. Let's use 20 features per observation
    data = loadtxt(infile, delimiter=",", usecols=range(20))
    labels = loadtxt(infile, delimiter=",", usecols=range(20, 21))
    labels.shape = (labels.size, 1)  # must be a 2d array
    tresult = talgo.compute(data, labels)

    # Now let's do some prediction
    # It runs only on a single node
    if d4p.my_procid() == 0:
        palgo = d4p.multinomial_naive_bayes_prediction(20)
        # read test data (with same #features)
        pdata = loadtxt(
            data_path / "naivebayes_test_dense.csv", delimiter=",", usecols=range(20)
        )
        # now predict using the model from the training above
        presult = palgo.compute(pdata, tresult.model)

        # Prediction result provides prediction
        assert presult.prediction.shape == (pdata.shape[0], 1)


if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()
    main()
    print("All looks good!")
    d4p.daalfini()

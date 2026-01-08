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

# daal4py association rules example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "apriori.csv"

    # configure a association_rules object
    algo = d4p.association_rules(discoverRules=True, minSupport=0.001, minConfidence=0.7)

    # let's provide a file directly, not a table/array
    result1 = algo.compute(str(infile))

    # We can also load the data ourselves and provide the numpy array
    data = readcsv(infile)
    result2 = algo.compute(data)

    # association_rules result objects provide antecedentItemsets,
    # confidence, consequentItemsets, largeItemsets and largeItemsetsSupport
    assert np.allclose(result1.largeItemsets, result2.largeItemsets)
    assert np.allclose(result1.largeItemsetsSupport, result2.largeItemsetsSupport)
    assert np.allclose(result1.antecedentItemsets, result2.antecedentItemsets)
    assert np.allclose(result1.consequentItemsets, result2.consequentItemsets)
    assert np.allclose(result1.confidence, result2.confidence)

    return result1


if __name__ == "__main__":
    result1 = main()
    print("Confidence: (20 first)")
    print(result1.confidence[0:20])
    print("All looks good!")

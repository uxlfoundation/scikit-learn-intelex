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

# daal4py PCA example for shared memory systems

from pathlib import Path

import numpy as np
from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / "data" / "batch"
    infile = data_path / "pca_normalized.csv"

    # 'normalization' is an optional parameter to PCA;
    # we use z-score which could be configured differently
    zscore = d4p.normalization_zscore()
    # configure a PCA object
    algo = d4p.pca(
        resultsToCompute="mean|variance|eigenvalue",
        isDeterministic=True,
        normalization=zscore,
    )

    # let's provide a file directly, not a table/array
    result1 = algo.compute(str(infile))

    # We can also load the data ourselves and provide the numpy array
    data = readcsv(infile)
    result2 = algo.compute(data)

    # PCA result objects provide eigenvalues, eigenvectors, means and variances
    assert np.allclose(result1.eigenvalues, result2.eigenvalues)
    assert np.allclose(result1.eigenvectors, result2.eigenvectors)
    assert np.allclose(result1.means, result2.means)
    assert np.allclose(result1.variances, result2.variances)
    assert result1.eigenvalues.shape == (1, data.shape[1])
    assert result1.eigenvectors.shape == (data.shape[1], data.shape[1])
    assert result1.means.shape == (1, data.shape[1])
    assert result1.variances.shape == (1, data.shape[1])

    return result1


if __name__ == "__main__":
    result1 = main()
    print("\nEigenvalues:\n", result1.eigenvalues)
    print("\nEigenvectors:\n", result1.eigenvectors)
    print("\nMeans:\n", result1.means)
    print("\nVariances:\n", result1.variances)
    print("All looks good!")

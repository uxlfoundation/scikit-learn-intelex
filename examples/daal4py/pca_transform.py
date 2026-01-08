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

from readcsv import pd_read_csv

import daal4py as d4p


def main(readcsv=pd_read_csv):
    data_path = Path(__file__).parent / "data" / "batch"
    data_file = data_path / "pca_transform.csv"
    nComponents = 2

    # read data
    data = readcsv(data_file, range(3))

    # configure a PCA object and perform PCA
    pca_algo = d4p.pca(isDeterministic=True, resultsToCompute="mean|variance|eigenvalue")
    pca_res = pca_algo.compute(data)

    # Apply transform with whitening because means and eigenvalues are provided
    pcatrans_algo = d4p.pca_transform(nComponents=nComponents)
    pcatrans_res = pcatrans_algo.compute(
        data, pca_res.eigenvectors, pca_res.dataForTransform
    )
    # pca_transform_result objects provides transformedData

    return (pca_res, pcatrans_res)


if __name__ == "__main__":
    pca_res, pcatrans_res = main()

    # print PCA results
    print("\nEigenvalues:\n", pca_res.eigenvalues)
    print("\nEigenvectors:\n", pca_res.eigenvectors)
    print("\nEigenvalues kv:\n", pca_res.dataForTransform["eigenvalue"])
    print("\nMeans kv:\n", pca_res.dataForTransform["mean"])
    print("\nVariances kv:\n", pca_res.dataForTransform["variance"])
    # print results of transform
    print("\nTransformed data:", pcatrans_res.transformedData)
    print("All looks good!")

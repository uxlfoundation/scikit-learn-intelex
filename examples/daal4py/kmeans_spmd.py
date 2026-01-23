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

# daal4py K-Means example for distributed memory systems; SPMD mode
# run like this:
#    mpirun -n 4 python ./kmeans_spmd.py

from pathlib import Path

from numpy import loadtxt

import daal4py as d4p


def main(method="plusPlusDense"):
    data_path = Path(__file__).parent / "data" / "distributed"
    infile = data_path / "kmeans_dense.csv"
    nClusters = 10
    maxIter = 25

    # configure a kmeans-init
    init_algo = d4p.kmeans_init(nClusters, method=method, distributed=True)
    # Load the data
    data = loadtxt(infile, delimiter=",")
    # now slice the data,
    # it would have been better to read only what we need, of course...
    rpp = int(data.shape[0] / d4p.num_procs())
    data = data[rpp * d4p.my_procid() : rpp * d4p.my_procid() + rpp, :]

    # compute initial centroids
    init_result = init_algo.compute(data)
    # The results provides the initial centroids
    assert init_result.centroids.shape[0] == nClusters

    # configure kmeans main object
    algo = d4p.kmeans(nClusters, maxIter, distributed=True)
    # compute the clusters/centroids
    result = algo.compute(data, init_result.centroids)

    # Note: we could have done this in just one line:
    # d4p.kmeans(nClusters, maxIter, assignFlag=True, distributed=True).compute(
    #     data,
    #     d4p.kmeans_init(
    #         nClusters,
    #         method="plusPlusDense",
    #         distributed=True
    #     ).compute(data).centroids
    # )

    # Kmeans result objects provide centroids, goalFunction,
    # nIterations and objectiveFunction
    assert result.centroids.shape[0] == nClusters
    assert result.nIterations <= maxIter
    # we need an extra call to kmeans to get the assignments
    # (not directly supported through parameter assignFlag yet in SPMD mode)
    algo = d4p.kmeans(nClusters, 0, assignFlag=True)
    # maxIt=0; not distributed, we compute on local data only!
    assignments = algo.compute(data, result.centroids).assignments

    return (assignments, result)


if __name__ == "__main__":
    # Initialize SPMD mode
    d4p.daalinit()
    assignments, result = main()
    # result is available on all processes - but we print only on root
    if d4p.my_procid() == 0:
        print("\nFirst 10 cluster assignments:\n", assignments[0:10])
        print("\nFirst 10 dimensions of centroids:\n", result.centroids[:, 0:10])
        print("\nObjective function value:\n", result.objectiveFunction)
        print("All looks good!")
    d4p.daalfini()

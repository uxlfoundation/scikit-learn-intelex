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

import os
import unittest
from pathlib import Path

import numpy as np
from test_daal4py_examples import (
    Base,
    Config,
    batch_data_path,
    distributed_data_path,
    example_path,
    import_module_any_path,
    low_order_moms_getter,
    readcsv,
)

import daal4py as d4p

# the examples are supposed to be executed in parallel with mpirun
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

missing_dist_reason = "daal4py was built without SPMD support"

is_parallel_execution = MPI is not None and MPI.COMM_WORLD.size > 1
parallel_reason = "Not running in distributed mode"


class SpmdDaal4pyBase(Base):
    @unittest.skipUnless(d4p.__has_dist__, missing_dist_reason)
    @unittest.skipUnless(is_parallel_execution, parallel_reason)
    def test_svd_spmd(self):
        ex = import_module_any_path(example_path / "svd_spmd")

        data, intermediate = self.call_main(ex)
        result = (
            np.matmul(
                np.matmul(
                    intermediate.leftSingularMatrix,
                    np.diag(intermediate.singularValues[0]),
                ),
                intermediate.rightSingularMatrix,
            ),
        )

        np.testing.assert_allclose(data, np.squeeze(result, axis=0))

    @unittest.skipUnless(d4p.__has_dist__, missing_dist_reason)
    @unittest.skipUnless(is_parallel_execution, parallel_reason)
    def test_qr_spmd(self):
        ex = import_module_any_path(example_path / "qr_spmd")

        data, intermediate = self.call_main(ex)
        result = np.matmul(intermediate.matrixQ, intermediate.matrixR)

        np.testing.assert_allclose(data, result)

    @unittest.skipUnless(d4p.__has_dist__, missing_dist_reason)
    @unittest.skipUnless(is_parallel_execution, parallel_reason)
    def test_kmeans_spmd(self):
        nClusters = 10
        maxIter = 25

        data = np.loadtxt(distributed_data_path / "kmeans_dense.csv", delimiter=",")

        rpp = int(data.shape[0] / d4p.num_procs())
        spmd_data = data[rpp * d4p.my_procid() : rpp * d4p.my_procid() + rpp, :]

        for init_method in [
            "plusPlusDense",
            "parallelPlusDense",
            "deterministicDense",
        ]:
            batch_init_res = d4p.kmeans_init(
                nClusters=nClusters, method=init_method
            ).compute(data)
            spmd_init_res = d4p.kmeans_init(
                nClusters=nClusters, method=init_method, distributed=True
            ).compute(spmd_data)

            if init_method in ["parallelPlusDense"]:
                print(
                    "Warning: It is well known "
                    "that results of parallelPlusDense init "
                    "does not match with batch algorithm"
                )
            else:
                reason = f"Initial centroids with {init_method} does not match with batch algorithm"
                np.testing.assert_allclose(
                    batch_init_res.centroids,
                    spmd_init_res.centroids,
                    err_msg=reason,
                )

            batch_res = d4p.kmeans(nClusters=nClusters, maxIterations=maxIter).compute(
                data, batch_init_res.centroids
            )
            spmd_res = d4p.kmeans(
                nClusters=nClusters, maxIterations=maxIter, distributed=True
            ).compute(spmd_data, spmd_init_res.centroids)

            if init_method in ["parallelPlusDense"]:
                print(
                    "Warning: It is well known "
                    "that results of parallelPlusDense init "
                    "does not match with batch algorithm"
                )
            else:
                reason = f"Final centroids with {init_method} does not match with batch algorithm"
                np.testing.assert_allclose(
                    batch_res.centroids, spmd_res.centroids, err_msg=reason
                )

    @unittest.skipUnless(d4p.__has_dist__, missing_dist_reason)
    @unittest.skipUnless(is_parallel_execution, parallel_reason)
    def test_dbscan_spmd(self):
        epsilon = 0.04
        minObservations = 45
        data = readcsv.np_read_csv(batch_data_path / "dbscan_dense.csv")

        batch_algo = d4p.dbscan(
            minObservations=minObservations,
            epsilon=epsilon,
            resultsToCompute="computeCoreIndices",
        )
        batch_result = batch_algo.compute(data)

        rpp = int(data.shape[0] / d4p.num_procs())
        node_stride = rpp * d4p.my_procid()
        node_range = range(node_stride, node_stride + rpp)
        node_data = data[node_range, :]

        spmd_algo = d4p.dbscan(
            minObservations=minObservations, epsilon=epsilon, distributed=True
        )
        spmd_result = spmd_algo.compute(node_data)

        # clusters can get different indexes in batch and spmd algos,
        # to compare them we should take care about it
        cluster_index_dict = {}
        for i in node_range:
            # border points assignments can be different
            # with different amount of nodes but cores are the same
            if i in batch_result.coreIndices:
                right = spmd_result.assignments[i - node_stride][0]
                if not batch_result.assignments[i][0] in cluster_index_dict:
                    cluster_index_dict[batch_result.assignments[i][0]] = right
                left = cluster_index_dict[batch_result.assignments[i][0]]
                self.assertEqual(left, right)


examples = [
    Config("covariance_spmd", "covariance.csv", "covariance"),
    Config("low_order_moms_spmd", "low_order_moms_dense.csv", low_order_moms_getter),
]

module_names_with_configs = [cfg.module_name for cfg in examples]

# add all examples that do not have an explicit config
for fname in os.listdir(example_path):
    if fname == "__init__.py":
        continue
    if not fname.endswith(".py"):
        continue
    if "spmd" not in fname:
        # here we only run spmd examples
        continue
    if "spmd_chunks_read" in fname:
        continue
    stem = Path(fname).stem
    if stem in module_names_with_configs:
        continue

    examples.append(Config(stem))

for cfg in examples:
    SpmdDaal4pyBase.add_test(cfg)


class Test(SpmdDaal4pyBase, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d4p.daalinit()

    @classmethod
    def tearDownClass(cls):
        d4p.daalfini()

    def call_main(self, ex):
        if not d4p.__has_dist__:
            self.skipTest(missing_dist_reason)
        if not is_parallel_execution:
            self.skipTest(parallel_reason)
        return ex.main()


if __name__ == "__main__":
    unittest.main()

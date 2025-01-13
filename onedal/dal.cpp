/*******************************************************************************
* Copyright 2023 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "onedal/common.hpp"
#include "onedal/version.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

/* common */
#ifdef ONEDAL_DATA_PARALLEL_SPMD
ONEDAL_PY_INIT_MODULE(spmd_policy);

/* algorithms */
ONEDAL_PY_INIT_MODULE(covariance);
ONEDAL_PY_INIT_MODULE(dbscan);
ONEDAL_PY_INIT_MODULE(ensemble);
ONEDAL_PY_INIT_MODULE(decomposition);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
ONEDAL_PY_INIT_MODULE(basic_statistics);
ONEDAL_PY_INIT_MODULE(linear_model);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
ONEDAL_PY_INIT_MODULE(kmeans_init);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
ONEDAL_PY_INIT_MODULE(kmeans);
ONEDAL_PY_INIT_MODULE(kmeans_common);
ONEDAL_PY_INIT_MODULE(neighbors);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
ONEDAL_PY_INIT_MODULE(logistic_regression);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
#else // ONEDAL_DATA_PARALLEL_SPMD
#ifdef ONEDAL_DATA_PARALLEL
ONEDAL_PY_INIT_MODULE(sycl);
#endif // ONEDAL_DATA_PARALLEL

ONEDAL_PY_INIT_MODULE(policy);
/* datatypes*/
ONEDAL_PY_INIT_MODULE(table);
ONEDAL_PY_INIT_MODULE(table_metadata);

/* primitives */
ONEDAL_PY_INIT_MODULE(get_tree);
ONEDAL_PY_INIT_MODULE(linear_kernel);
ONEDAL_PY_INIT_MODULE(rbf_kernel);
ONEDAL_PY_INIT_MODULE(polynomial_kernel);
ONEDAL_PY_INIT_MODULE(sigmoid_kernel);

/* algorithms */
ONEDAL_PY_INIT_MODULE(covariance);
ONEDAL_PY_INIT_MODULE(dbscan);
ONEDAL_PY_INIT_MODULE(ensemble);
ONEDAL_PY_INIT_MODULE(decomposition);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
ONEDAL_PY_INIT_MODULE(basic_statistics);
ONEDAL_PY_INIT_MODULE(linear_model);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
ONEDAL_PY_INIT_MODULE(kmeans_init);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
ONEDAL_PY_INIT_MODULE(kmeans);
ONEDAL_PY_INIT_MODULE(kmeans_common);
ONEDAL_PY_INIT_MODULE(neighbors);
ONEDAL_PY_INIT_MODULE(svm);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
ONEDAL_PY_INIT_MODULE(logistic_regression);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
ONEDAL_PY_INIT_MODULE(finiteness_checker);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
#endif // ONEDAL_DATA_PARALLEL_SPMD

#ifdef ONEDAL_DATA_PARALLEL_SPMD
PYBIND11_MODULE(_onedal_py_spmd_dpc, m) {
    init_spmd_policy(m);

    init_covariance(m);
    init_dbscan(m);
    init_decomposition(m);
    init_ensemble(m);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
    init_basic_statistics(m);
    init_linear_model(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
    init_kmeans_init(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
    init_kmeans(m);
    init_kmeans_common(m);
    init_neighbors(m);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
    init_logistic_regression(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
}
#else
#ifdef ONEDAL_DATA_PARALLEL
PYBIND11_MODULE(_onedal_py_dpc, m) {
    init_sycl(m);
#else
PYBIND11_MODULE(_onedal_py_host, m) {
#endif
    init_policy(m);
    init_table(m);
    init_table_metadata(m);

    init_linear_kernel(m);
    init_rbf_kernel(m);
    init_polynomial_kernel(m);
    init_sigmoid_kernel(m);
    init_get_tree(m);

    init_covariance(m);
    init_dbscan(m);
    init_decomposition(m);
    init_ensemble(m);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
    init_basic_statistics(m);
    init_linear_model(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230100
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
    init_kmeans_init(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20230200
    init_kmeans(m);
    init_kmeans_common(m);
    init_neighbors(m);
    init_svm(m);
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
    init_logistic_regression(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240001
#if defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
    init_finiteness_checker(m);
#endif // defined(ONEDAL_VERSION) && ONEDAL_VERSION >= 20240700
}
#endif // ONEDAL_DATA_PARALLEL_SPMD

} // namespace oneapi::dal::python

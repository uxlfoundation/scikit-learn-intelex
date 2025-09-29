/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "onedal/primitives/pairwise_distances.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

ONEDAL_PY_DECLARE_INSTANTIATOR(init_distance_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_distance_compute_ops);

ONEDAL_PY_INIT_MODULE(cosine_distance) {
    using namespace dal::detail;
    using namespace cosine_distance;
    using input_t = compute_input<task::compute>;
    using result_t = compute_result<task::compute>;
    using param2desc_t = distance_params2desc<descriptor>;

    auto sub = m.def_submodule("cosine_distance");
#ifndef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_distance_result, sub, result_t);
    ONEDAL_PY_INSTANTIATE(init_distance_compute_ops,
                          sub,
                          policy_list,
                          input_t,
                          result_t,
                          param2desc_t,
                          method::dense);
#endif
}

} // namespace oneapi::dal::python

/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#include "onedal/primitives/kernel_functions.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

ONEDAL_PY_DECLARE_INSTANTIATOR(init_kernel_result);
ONEDAL_PY_DECLARE_INSTANTIATOR(init_kernel_compute_ops);

ONEDAL_PY_INIT_MODULE(rbf_kernel) {
    using namespace dal::detail;
    using namespace rbf_kernel;
    using input_t = compute_input<task::compute>;
    using result_t = compute_result<task::compute>;
    using param2desc_t = kernel_params2desc<descriptor>;

    auto sub = m.def_submodule("rbf_kernel");
#ifndef ONEDAL_DATA_PARALLEL_SPMD
    ONEDAL_PY_INSTANTIATE(init_kernel_result, sub, result_t);
    ONEDAL_PY_INSTANTIATE(init_kernel_compute_ops,
                          sub,
                          policy_list,
                          input_t,
                          result_t,
                          param2desc_t,
                          method::dense);
#endif
}

} // namespace oneapi::dal::python

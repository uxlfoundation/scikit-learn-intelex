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

#include "oneapi/dal/detail/policy.hpp"
#include "onedal/common/policy.hpp"
#include "onedal/common/pybind11_helpers.hpp"

#ifdef ONEDAL_DATA_PARALLEL_SPMD
#include "oneapi/dal/detail/spmd_policy.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"
#endif // ONEDAL_DATA_PARALLEL_SPMD

namespace py = pybind11;

namespace oneapi::dal::python {

using host_policy_t = dal::detail::host_policy;
using default_host_policy_t = dal::detail::default_host_policy;

void instantiate_host_policy(py::module& m) {
    constexpr const char name[] = "host_policy";
    py::class_<host_policy_t> policy(m, name);
    policy.def(py::init<host_policy_t>());
    instantiate_host_policy(policy);
}

void instantiate_default_host_policy(py::module& m) {
    constexpr const char name[] = "default_host_policy";
    py::class_<default_host_policy_t> policy(m, name);
    policy.def(py::init<default_host_policy_t>());
    instantiate_host_policy(policy);
}

#ifdef ONEDAL_DATA_PARALLEL

dp_policy_t make_dp_policy(std::uint32_t id) {
    sycl::queue queue = get_queue_by_device_id(id);
    return dp_policy_t{ std::move(queue) };
}

dp_policy_t make_dp_policy(const py::object& syclobj) {
    sycl::queue queue = get_queue_from_python(syclobj);
    return dp_policy_t{ std::move(queue) };
}

dp_policy_t make_dp_policy(const std::string& filter) {
    sycl::queue queue = get_queue_by_filter_string(filter);
    return dp_policy_t{ std::move(queue) };
}

void instantiate_data_parallel_policy(py::module& m) {
    constexpr const char name[] = "data_parallel_policy";
    py::class_<dp_policy_t> policy(m, name);
    policy.def(py::init<dp_policy_t>());
    policy.def(py::init<const sycl::queue&>());
    policy.def(py::init([](std::uint32_t id) {
        return make_dp_policy(id);
    }));
    policy.def(py::init([](const std::string& filter) {
        return make_dp_policy(filter);
    }));
    policy.def(py::init([](const py::object& syclobj) {
        return make_dp_policy(syclobj);
    }));
    policy.def("get_device_id", [](const dp_policy_t& policy) {
        return get_device_id(policy);
    });
    policy.def("get_device_name", [](const dp_policy_t& policy) {
        return get_device_name(policy);
    });
    m.def("get_used_memory", &get_used_memory, py::return_value_policy::take_ownership);
}
#endif // ONEDAL_DATA_PARALLEL
#ifdef ONEDAL_DATA_PARALLEL_SPMD
using dp_policy_t = dal::detail::data_parallel_policy;
using spmd_policy_t = dal::detail::spmd_policy<dp_policy_t>;

inline spmd_policy_t make_spmd_policy(dp_policy_t&& local) {
    sycl::queue& queue = local.get_queue();
    using backend_t = dal::preview::spmd::backend::mpi;
    auto comm = dal::preview::spmd::make_communicator<backend_t>(queue);
    return spmd_policy_t{ std::forward<dp_policy_t>(local), std::move(comm) };
}

template <typename... Args>
inline spmd_policy_t make_spmd_policy(Args&&... args) {
    auto local = make_dp_policy(std::forward<Args>(args)...);
    return make_spmd_policy(std::move(local));
}

template <typename Arg, typename Policy = spmd_policy_t>
inline void instantiate_costructor(py::class_<Policy>& policy) {
    policy.def(py::init([](const Arg& arg) {
        return make_spmd_policy(arg);
    }));
}

void instantiate_spmd_policy(py::module& m) {
    constexpr const char name[] = "spmd_data_parallel_policy";
    py::class_<spmd_policy_t> policy(m, name);
    policy.def(py::init<spmd_policy_t>());
    policy.def(py::init([](const dp_policy_t& local) {
        return make_spmd_policy(local);
    }));
    policy.def(py::init([](std::uint32_t id) {
        return make_spmd_policy(id);
    }));
    policy.def(py::init([](const std::string& filter) {
        return make_spmd_policy(filter);
    }));
    policy.def(py::init([](const py::object& syclobj) {
        return make_spmd_policy(syclobj);
    }));
    policy.def("get_device_id", [](const spmd_policy_t& policy) {
        return get_device_id(policy.get_local());
    });
    policy.def("get_device_name", [](const spmd_policy_t& policy) {
        return get_device_name(policy.get_local());
    });
}
#endif // ONEDAL_DATA_PARALLEL_SPMD

py::object get_policy(py::object obj) {
    if (!obj.is(py::none())) {
#ifdef ONEDAL_DATA_PARALLEL_SPMD
        return py::type::of<spmd_policy_t>()(obj);
#elif ONEDAL_DATA_PARALLEL
        return py::type::of<dp_policy_t>()(obj);
#else
        throw std::invalid_argument("queues are not supported in the oneDAL backend");
#endif // ONEDAL_DATA_PARALLEL
    }
    return py::type::of<host_policy_t>()();
};

ONEDAL_PY_INIT_MODULE(policy) {
    instantiate_host_policy(m);
    instantiate_default_host_policy(m);
#ifdef ONEDAL_DATA_PARALLEL
    instantiate_data_parallel_policy(m);
#endif // ONEDAL_DATA_PARALLEL
#ifdef ONEDAL_DATA_PARALLEL_SPMD
    instantiate_spmd_policy(m);
#endif // ONEDAL_DATA_PARALLEL_SPMD
    m.def("get_policy", &get_policy, py::arg("queue") = py::none());
}
} // namespace oneapi::dal::python

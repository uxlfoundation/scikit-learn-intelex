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
// NOTE: only the communicator *type* is needed here (device_memory_access tags).
// The concrete transport backend (MPI, and later CCL) lives in a separate
// pybind11 module (``_onedal_spmd_mpi``) so that this algorithm module does not
// link against MPI. The communicator instance is created there and passed in.
#include "oneapi/dal/spmd/communicator.hpp"
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
}
#endif // ONEDAL_DATA_PARALLEL
#ifdef ONEDAL_DATA_PARALLEL_SPMD
namespace spmd = dal::preview::spmd;

// Communicator types matching the two SPMD policies. Instances are produced by
// the standalone ``_onedal_spmd_mpi`` module (and later ``_onedal_spmd_ccl``);
// this module only consumes them, so it carries no transport dependency.
using host_comm_t = spmd::communicator<spmd::device_memory_access::none>;
using device_comm_t = spmd::communicator<spmd::device_memory_access::usm>;

using spmd_host_policy_t = dal::detail::spmd_policy<host_policy_t>;
using spmd_dp_policy_t = dal::detail::spmd_policy<dal::detail::data_parallel_policy>;

// Host (CPU-distributed) SPMD policy: built purely from a communicator, no
// queue required. This is what enables dask-mpi-style CPU clusters.
void instantiate_spmd_host_policy(py::module& m) {
    constexpr const char name[] = "spmd_host_policy";
    py::class_<spmd_host_policy_t> policy(m, name);
    policy.def(py::init<spmd_host_policy_t>());
    policy.def(py::init([](const host_comm_t& comm) {
        return spmd_host_policy_t{ host_policy_t{}, comm };
    }));
    policy.def("get_device_id", [](const spmd_host_policy_t&) -> std::uint32_t {
        return std::uint32_t{ 0u };
    });
    policy.def("get_device_name", [](const spmd_host_policy_t&) -> std::string {
        return std::string{ "cpu" };
    });
}

// Device (GPU-distributed) SPMD policy: built from a local data-parallel policy
// (i.e. a SYCL queue) plus a device communicator.
void instantiate_spmd_data_parallel_policy(py::module& m) {
    constexpr const char name[] = "spmd_data_parallel_policy";
    py::class_<spmd_dp_policy_t> policy(m, name);
    policy.def(py::init<spmd_dp_policy_t>());
    policy.def(py::init([](const dp_policy_t& local, const device_comm_t& comm) {
        return spmd_dp_policy_t{ local, comm };
    }));
    policy.def(py::init([](const py::object& syclobj, const device_comm_t& comm) {
        return spmd_dp_policy_t{ make_dp_policy(syclobj), comm };
    }));
    policy.def("get_device_id", [](const spmd_dp_policy_t& policy) {
        return get_device_id(policy.get_local());
    });
    policy.def("get_device_name", [](const spmd_dp_policy_t& policy) {
        return get_device_name(policy.get_local());
    });
}
#endif // ONEDAL_DATA_PARALLEL_SPMD

#ifdef ONEDAL_DATA_PARALLEL_SPMD
// SPMD policies now require a communicator, which is created in the standalone
// ``_onedal_spmd_mpi`` module. Construction therefore happens through the
// exposed ``spmd_host_policy`` / ``spmd_data_parallel_policy`` constructors
// (wired up by the Python orchestration layer), not through this helper. This
// stub keeps the ``get_policy`` symbol available but makes the requirement to
// pass a communicator explicit rather than failing with an opaque overload
// error.
py::object get_policy(py::object) {
    throw std::invalid_argument(
        "SPMD policies require a communicator: construct spmd_host_policy(comm) "
        "or spmd_data_parallel_policy(queue, comm) using a communicator from the "
        "_onedal_spmd_mpi module.");
}
#else
py::object get_policy(py::object obj) {
    if (!obj.is(py::none())) {
#ifdef ONEDAL_DATA_PARALLEL
        return py::type::of<dp_policy_t>()(obj);
#else
        throw std::invalid_argument("queues are not supported in the oneDAL backend");
#endif // ONEDAL_DATA_PARALLEL
    }
    return py::type::of<host_policy_t>()();
};
#endif // ONEDAL_DATA_PARALLEL_SPMD

ONEDAL_PY_INIT_MODULE(policy) {
#ifdef ONEDAL_DATA_PARALLEL_SPMD
    instantiate_spmd_host_policy(m);
    instantiate_spmd_data_parallel_policy(m);
#else
    instantiate_host_policy(m);
    instantiate_default_host_policy(m);
#ifdef ONEDAL_DATA_PARALLEL
    instantiate_data_parallel_policy(m);
#endif // ONEDAL_DATA_PARALLEL
#endif // ONEDAL_DATA_PARALLEL_SPMD
    m.def("get_policy", &get_policy, py::arg("queue") = py::none());
}
} // namespace oneapi::dal::python

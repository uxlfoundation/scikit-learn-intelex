/*******************************************************************************
* Copyright 2024 Intel Corporation
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

#pragma once

#include <string>
#include <cstdint>
#include <optional>

#ifdef ONEDAL_DATA_PARALLEL
#include <sycl/sycl.hpp>
#endif // ONEDAL_DATA_PARALLEL

#include <pybind11/pybind11.h>

#include "oneapi/dal/detail/policy.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

#ifdef ONEDAL_DATA_PARALLEL

std::optional<sycl::device> get_device_by_id(std::uint32_t id);
std::optional<std::uint32_t> get_device_id(const sycl::device& device);

sycl::queue extract_queue(py::capsule capsule);
sycl::context extract_context(py::capsule capsule);
sycl::queue extract_from_capsule(py::capsule capsule);

sycl::queue get_queue_by_get_capsule(const py::object& syclobj);
sycl::queue get_queue_by_pylong_pointer(const py::int_& syclobj);
sycl::queue get_queue_by_filter_string(const std::string& filter);
sycl::queue get_queue_by_device_id(std::uint32_t id);
sycl::queue get_queue_from_python(const py::object& syclobj);

using dp_policy_t = detail::data_parallel_policy;

std::uint32_t get_device_id(const sycl::queue& queue);
std::uint32_t get_device_id(const dp_policy_t& policy);
std::size_t get_used_memory(const py::object& syclobj);
std::string get_device_name(const dp_policy_t& policy);
std::string get_device_name(const sycl::device& device);

/// TODO: This is a workaround class.
/// It hides deprecated ``sycl::ext::oneapi::filter_selector`` to get rid of build warnings
/// until a better solution is provided.
struct filter_selector_wrapper {
    filter_selector_wrapper(std::string filter) : filter_selector_{ filter } {}

    int operator()(const sycl::device& dev) {
        return filter_selector_(dev);
    }

private:
    sycl::ext::oneapi::filter_selector filter_selector_;
};

py::capsule pack_queue(const std::shared_ptr<sycl::queue>& queue);

#endif // ONEDAL_DATA_PARALLEL

} // namespace oneapi::dal::python

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

#include "onedal/datatypes/dlpack/device_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::dlpack {

constexpr inline auto cpu = DLDeviceType::kDLCPU;
constexpr inline auto oneapi = DLDeviceType::kDLOneAPI;

DLDevice get_cpu_device() {
    return DLDevice{ cpu, 0 };
}

bool is_cpu_device(DLDevice device) {
    const bool is_trivial = device.device_id == 0;
    const bool is_dlpack_cpu = device.device_type == cpu;
    return is_dlpack_cpu && is_trivial;
}

bool is_oneapi_device(DLDevice device) {
    const bool trivial = device.device_id == 0;
    const bool is_dlpack_oneapi = device.device_type == oneapi;
#ifdef ONEDAL_DATA_PARALLEL
    auto dev_opt = get_device_by_id(device.device_id);
    const bool is_known_by_id = dev_opt.has_value();
#else // ONEDAL_DATA_PARALLEL
    constexpr bool is_known_by_id = false;
#endif // ONEDAL_DATA_PARALLEL
    return is_dlpack_oneapi && is_known_by_id;
}

bool is_unknown_device(DLDevice device) {
    const bool dlpack_cpu = is_cpu_device(device);
    const bool dlpack_oneapi = is_oneapi_device(device);
    return !dlpack_cpu && !dlpack_oneapi;
}

#ifdef ONEDAL_DATA_PARALLEL

std::optional<sycl::device> convert_to_sycl(DLDevice device) {
    if (is_cpu_device(device)) {
        return sycl::ext::oneapi::detail::select_device( //
            &sycl::cpu_selector_v);
    }
    else if (is_oneapi_device(device)) {
        return get_device_by_id(device.device_id);
    }
    else {
        return {};
    }
}

std::optional<DLDevice> convert_from_sycl(sycl::device device) {
    if (auto id = get_device_id(device)) {
        const std::uint32_t uid = id.value();
        auto raw = static_cast<std::int32_t>(uid);
        return { DLDevice{ oneapi, raw } };
    }
    else {
        return {};
    }
}

#endif // ONEDAL_DATA_PARALLEL

py::object to_policy(DLDevice device) {
    if (is_cpu_device(device)) {
        detail::default_host_policy pol{};
        return py::cast(std::move(pol));
    }
#ifdef ONEDAL_DATA_PARALLEL
    else if (is_oneapi_device(device)) {
        auto dev = convert_to_sycl(device);
        sycl::queue queue{ dev.value() };
        detail::data_parallel_policy pol{ queue };
        return py::cast(std::move(pol));
    }
#endif // ONEDAL_DATA_PARALLEL
    else {
        throw std::runtime_error("Unknown device");
    }
}

DLDevice get_device(std::shared_ptr<sycl::queue> ptr) {
#ifdef ONEDAL_DATA_PARALLEL
    if (ptr.get() != nullptr) {
        const sycl::device dev = ptr->get_device();
        if (auto device = convert_from_sycl(dev)) {
            return device.value();
        }
    }
#endif // ONEDAL_DATA_PARALLEL
    return get_cpu_device();
}

std::shared_ptr<sycl::queue> get_queue(DLDevice device) {
    if (is_cpu_device(device)) {
        return { nullptr };
    }
#ifdef ONEDAL_DATA_PARALLEL
    if (is_oneapi_device(device)) {
        const auto dev = convert_to_sycl(device);
        const sycl::device& unwrapped = dev.value();
        return std::make_shared<sycl::queue>(unwrapped);
    }
#endif // ONEDAL_DATA_PARALLEL
    else {
        throw std::runtime_error("Unknown device");
        return { nullptr };
    }
}

py::object to_policy(py::tuple tp) {
    constexpr const char err[] = "Ill-formed device tuple";

    if (tp.size() != py::ssize_t{ 2ul }) {
        throw std::runtime_error(err);
    }

    auto type = tp[0ul].cast<std::int32_t>();

    DLDevice desc{ static_cast<DLDeviceType>(type), tp[1].cast<std::int32_t>() };

    return to_policy(std::move(desc));
}

void instantiate_convert_to_policy(py::module& pm) {
    pm.def("convert_to_policy", [](py::tuple tp) -> py::object {
        return to_policy(std::move(tp));
    });
}

} // namespace oneapi::dal::python::dlpack
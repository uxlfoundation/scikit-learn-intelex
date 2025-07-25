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

#include <pybind11/operators.h>

#include "onedal/common/sycl_interfaces.hpp"
#include "onedal/common/pybind11_helpers.hpp"

namespace py = pybind11;

namespace oneapi::dal::python {

void instantiate_sycl_interfaces(py::module& m) {
    // These classes mirror a subset of functionality of the dpctl python
    // package's `SyclQueue` and `SyclDevice` objects.  In the case that dpctl
    // is not installed, these classes will enable scikit-learn-intelex to still
    // properly offload to other devices when built with the dpc backend.
#ifdef ONEDAL_DATA_PARALLEL
    py::class_<sycl::queue> syclqueue(m, "SyclQueue");
    syclqueue.def(py::init<const sycl::device&>())
        .def(py::init([](const std::string& filter) {
            return get_queue_by_filter_string(filter);
        }))
        .def(py::init([](const py::int_& obj) {
            return get_queue_by_pylong_pointer(obj);
        }))
        .def(py::init([](const py::object& syclobj) {
            return get_queue_from_python(syclobj);
        }))
        .def("_get_capsule",
             [](const sycl::queue& queue) {
                 return pack_queue(std::make_shared<sycl::queue>(queue));
             })
        .def_property_readonly("sycl_device", &sycl::queue::get_device)
        .def(py::self == py::self)
        .def(py::self != py::self);

    // expose limited sycl device features to python for oneDAL analysis
    py::class_<sycl::device> sycldevice(m, "SyclDevice");
    sycldevice
        .def(py::init([](std::uint32_t id) {
            return get_device_by_id(id).value();
        }))
        .def_property_readonly("has_aspect_fp64",
                               [](const sycl::device& device) {
                                   return device.has(sycl::aspect::fp64);
                               })
        .def_property_readonly("has_aspect_fp16",
                               [](const sycl::device& device) {
                                   return device.has(sycl::aspect::fp16);
                               })
        .def_property_readonly("filter_string",
                               [](const sycl::device& device) {
                                   // assumes we are not working with accelerators
                                   // This is a minimal reproduction of DPCTL_GetRelativeDeviceId
                                   std::uint32_t outidx = 0;
                                   std::string filter = get_device_name(device);
                                   auto devtype =
                                       device.get_info<sycl::info::device::device_type>();
                                   auto devs = device.get_devices(devtype);
                                   auto be = device.get_platform().get_backend();
                                   for (std::uint32_t id = 0; devs[outidx] != device; ++id) {
                                       if (devs[id].get_platform().get_backend() == be)
                                           ++outidx;
                                   }
                                   return py::str(filter + ":") + py::str(py::int_(outidx));
                               })
        .def("get_device_id",
             [](const sycl::device& device) {
                 return get_device_id(device).value();
             })
        .def_property_readonly("is_cpu", &sycl::device::is_cpu)
        .def_property_readonly("is_gpu", &sycl::device::is_gpu)
        .def(py::self == py::self)
        .def(py::self != py::self);
#else
    struct syclqueue {};
    py::class_<syclqueue> syclqueue(m, "SyclQueue");
    // inspired from pybind11 PR#4698 which turns init into a no-op
    syclqueue
        .def(py::init([]() {
            return nullptr;
        }))
        .def_static("__new__", [](const py::object& cls, const py::object& obj) {
            // this object is defined for the host build, where SYCL support is not available.
            // This class acts as the failure point to target_offload, which will throw an
            // error in all circumstances if any value but the default value ("auto"), or a string
            // starting with "cpu". The returned "queue" is a None. Must be a class to work with
            // isinstance
            if (!py::isinstance<py::str>(obj) || obj.cast<std::string>() != "auto") {
                throw std::invalid_argument(
                    "device use via `target_offload` is only supported with the DPC++ backend");
            }
            return py::none();
        });
#endif
}

ONEDAL_PY_INIT_MODULE(sycl) {
    instantiate_sycl_interfaces(m);
}

} // namespace oneapi::dal::python

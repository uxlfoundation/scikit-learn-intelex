/*******************************************************************************
* Copyright Contributors to the oneDAL project
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

#include "oneapi/dal/array.hpp"

namespace oneapi::dal::python {

template <typename T>
dal::array<T> transfer_to_host(const dal::array<T> &array) {
#ifdef ONEDAL_DATA_PARALLEL
    auto opt_queue = array.get_queue();
    if (opt_queue.has_value()) {
        auto device = opt_queue->get_device();
        if (!device.is_cpu()) {
            const auto *device_data = array.get_data();

            auto memory_kind = sycl::get_pointer_type(device_data, opt_queue->get_context());
            if (memory_kind == sycl::usm::alloc::unknown) {
                throw std::runtime_error("[convert_to_numpy] Unknown memory type");
            }
            if (memory_kind == sycl::usm::alloc::device) {
                auto host_array = dal::array<T>::empty(array.get_count());
                opt_queue->memcpy(host_array.get_mutable_data(), device_data, array.get_size())
                    .wait_and_throw();
                return host_array;
            }
            if (memory_kind == sycl::usm::alloc::shared) {
                // if a shared allocation no movement is necessary but sync the data.
                opt_queue.wait();
                return array;
            }
        }
    }
#endif

    return array;
}

} // namespace oneapi::dal::python

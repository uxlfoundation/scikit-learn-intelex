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

#include <pybind11/pybind11.h>

#include "onedal/interop/utils/common.hpp"

#include "onedal/interop/dlpack/api/dlpack.h"
#include "onedal/interop/dlpack/dlpack_utils.hpp"
#include "onedal/interop/dlpack/dtype_conversion.hpp"
#include "onedal/interop/dlpack/dlpack_interface.hpp"
#include "onedal/interop/dlpack/device_conversion.hpp"

namespace oneapi::dal::python::interop::dlpack {

template <std::int64_t dim>
DLTensor produce_unmanaged(std::shared_ptr<dlpack_interface<dim>> ptr) {
    DLTensor result;

    result.byte_offset = ptr->offset;
    result.shape = ptr->shape.data();
    result.strides = ptr->strides.data();
    result.device = get_device(ptr->queue);
    result.dtype = convert_dal_to_dlpack_type(ptr->dtype);
    result.ndim = detail::integral_cast<std::int32_t>(dim);
    result.data = reinterpret_cast<void*>(ptr->data.second);

    return result;
}

template <std::int64_t dim, typename Deleter>
inline std::shared_ptr<dlpack_interface<dim>> convert(const DLManagedTensor& managed,
                                                      Deleter&& deleter) {
    auto* const ptr = new dlpack_interface<dim>{};
    const auto& tensor = managed.dl_tensor;

    ptr->data.second = true;
    ptr->queue = get_queue(tensor.device);
    ptr->dtype = convert_dlpack_to_dal_type(tensor.dtype);
    ptr->data.first = reinterpret_cast<std::uintptr_t>(tensor.data);

    if (tensor.ndim != static_cast<std::int32_t>(dim)) {
        throw std::runtime_error("Inconsistent dimensions");
    }

    for (std::int64_t d = 0l; d < dim; ++d) {
        ptr->shape.at(d) = tensor.shape[d];
    }

    if (tensor.strides == NULL) {
        ptr->strides = utils::get_c_strides(ptr->shape);
    }
    else {
        for (std::int64_t d = 0l; d < dim; ++d) {
            ptr->strides.at(d) = tensor.strides[d];
        }
    }

    return std::shared_ptr<dlpack_interface<dim>>( //
        ptr,
        std::forward<Deleter>(deleter));
}

template <std::int64_t dim>
std::shared_ptr<dlpack_interface<dim>> get_dlpack_interface(py::capsule capsule) {
    static const char new_name[] = "used_dltensor";

    capsule.inc_ref();
    capsule.set_name(new_name);
    const auto& ref = *capsule.get_pointer<DLManagedTensor>();

    auto deleter = [capsule](auto* ptr) {
        capsule.dec_ref();
    };

    return convert<dim>(ref, std::move(deleter));
}

#define INSTANTIATE_DIM(DIM)                                                     \
    template DLTensor produce_unmanaged(std::shared_ptr<dlpack_interface<DIM>>); \
    template std::shared_ptr<dlpack_interface<DIM>> get_dlpack_interface<DIM>(py::capsule);

INSTANTIATE_DIM(1)
INSTANTIATE_DIM(2)

} // namespace oneapi::dal::python::interop::dlpack

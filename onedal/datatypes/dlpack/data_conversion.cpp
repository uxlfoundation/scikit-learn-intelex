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

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

#include "onedal/datatypes/dlpack/data_conversion.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "onedal/common/sycl_interfaces.hpp"
#endif // ONEDAL_DATA_PARALLEL

namespace oneapi::dal::python::dlpack {

template <typename T, typename managed_t>
inline dal::homogen_table convert_to_homogen_impl(managed_t* dlm_tensor, py::object q_obj) {
    dal::homogen_table res{};

    DLTensor tensor = dlm_tensor->dl_tensor;

    // if a nullptr, return an empty.
    if (!tensor.data) {
        return res;
    }

    // Get pointer to the data following dlpack.h conventions.
    // use static cast because data is known to be void*, and reintrepret_cast
    // since we know that T* is of a set specific types that we will guarantee
    // for the compiler.
    const char* raw_ptr = static_cast<const char*>(tensor.data);
    const T* ptr = reinterpret_cast<const T*>(raw_ptr + tensor.byte_offset);

    // get shape, if 0 or 1 dimensional, force col count to at least 1
    // this will force the output dal table to be 2d
    const std::int64_t row_count = tensor.shape[0];
    const std::int64_t col_count = get_ndim(tensor) > 1 ? tensor.shape[1] : 1l;

    // get data layout for homogeneous check
    const dal::data_layout layout = get_dlpack_layout(tensor);

    // create the dlpack deleter, which requires calling the deleter from the managed DL tensor.
    // This must be done instead of a decref, as the data doesn't necessarily come from python,
    // It is expected that deleter handles memory cleanup (including possible python decrefs)
    const auto deleter = [dlm_tensor](const T* data) {
        if (dlm_tensor->deleter != nullptr) {
            dlm_tensor->deleter(dlm_tensor);
        }
    };

    // check_dlpack_oneAPI_device will check if data is on a oneAPI device. If it is on an
    // unsupported device it will throw an error.
    if (check_dlpack_oneAPI_device(tensor.device.device_type)) {
#ifdef ONEDAL_DATA_PARALLEL
        // if located on a SYCL device, use the queue. Generate queue from dlpack device information.
        // If a queue is given, it will override the general queue that would be generated
        // from get_queue_by_device_id. This is important as dlpack oneAPI devices are not aware of
        // sub-devices, priority queues or sycl contexts, which can cause the tensor only to be usable
        // in oneDAL when sycl default contexts and parent devices are used. It is assumed that the
        // external queue  is a more specific which properly handles these cases (allowing for oneDAL).
        sycl::queue queue;
        queue = !q_obj.is(py::none()) ? get_queue_from_python(q_obj)
                                      : get_queue_by_device_id(tensor.device.device_id);

        res = dal::homogen_table(queue,
                                 ptr,
                                 row_count,
                                 col_count,
                                 deleter,
                                 std::vector<sycl::event>{},
                                 layout);
#else
        throw std::invalid_argument(
            "Input array located on a oneAPI device, but sklearnex installation does not have SYCL support.");
#endif
    }
    else {
        res = dal::homogen_table(ptr, row_count, col_count, deleter, layout);
    }
    return res;
}

dal::table convert_to_table(py::object obj, py::object q_obj, bool recursed) {
    dal::table res;
    bool versioned;
    DLManagedTensor* dlm;
    DLManagedTensorVersioned* dlmv;
    DLTensor tensor;
    dal::data_type dtype;

    // extract __dlpack__ attribute from the input obj. This function should
    // only be called if the attribute has been checked.
    py::capsule caps = obj.attr("__dlpack__")();

    tensor = get_dlpack_tensor(caps, dlm, dlmv, versioned);

    // Extract and convert a DLpack data type into a oneDAL dtype.
    dtype = convert_dlpack_to_dal_type(tensor.dtype);

    // if there is a queue, check that the data matches the necessary precision.
#ifdef ONEDAL_DATA_PARALLEL
    if (!q_obj.is(py::none()) && !q_obj.attr("sycl_device").attr("has_aspect_fp64").cast<bool>() &&
        dtype == dal::data_type::float64) {
        // If the queue exists, doesn't have the fp64 aspect, and the data is float64
        // then cast it to float32 (using reduce_precision), error raised in reduce_precision
        if (!recursed) {
            py::object copy = reduce_precision(obj);
            res = convert_to_table(copy, q_obj, true);
        }
        else {
            throw std::invalid_argument("dlpack input could not be converted into onedal table.");
        }
        return res;
    }
#endif // ONEDAL_DATA_PARALLEL

    // unusual data format found, try to make contiguous, otherwise throw error
    // error throw located in regenerate_layout
    if (get_dlpack_layout(tensor) == dal::data_layout::unknown) {
        // NOTE: this attempts to make a contiguous deep copy of the data
        // if possible, this is expected to be a special case
        if (!recursed) {
            py::object copy = regenerate_layout(obj);
            res = convert_to_table(copy, q_obj, true);
        }
        else {
            throw std::invalid_argument("dlpack input could not be converted into onedal table.");
        }
        return res;
    }

#define MAKE_HOMOGEN_TABLE(CType)                                                           \
    res = versioned ? convert_to_homogen_impl<CType, DLManagedTensorVersioned>(dlmv, q_obj) \
                    : convert_to_homogen_impl<CType, DLManagedTensor>(dlm, q_obj);
    SET_CTYPE_FROM_DAL_TYPE(dtype,
                            MAKE_HOMOGEN_TABLE,
                            throw py::type_error("Found unsupported tensor type"));
#undef MAKE_HOMOGEN_TABLE

    // take ownership of the capsule, this is important to prevent data deletion
    dlpack_take_ownership(caps);
    return res;
}

DLDevice get_dlpack_device(const dal::array<byte_t>& array) {
    DLDevice device;
#ifdef ONEDAL_DATA_PARALLEL
    auto queue = array.get_queue();
    device = queue.has_value()
                 ? DLDevice{ kDLOneAPI, static_cast<std::int32_t>(get_device_id(queue.value())) }
                 : DLDevice{ kDLCPU, std::int32_t(0) };
#else
    device = DLDevice{ kDLCPU, std::int32_t(0) };
#endif //ONEDAL_DATA_PARALLEL
    return device;
}

DLDevice get_dlpack_device(const dal::table& input) {
    if (input.get_kind() == dal::homogen_table::kind()) {
        auto homogen_input = reinterpret_cast<const dal::homogen_table&>(input);
        dal::array<byte_t> array = dal::detail::get_original_data(homogen_input);
        return get_dlpack_device(array);
    }
    else {
        return DLDevice{ kDLCPU, std::int32_t(0) };
    }
}

DLTensor construct_dlpack_tensor(const dal::array<byte_t>& array,
                                 std::int64_t row_count,
                                 std::int64_t column_count,
                                 const dal::data_type& dtype,
                                 const dal::data_layout& layout) {
    DLTensor tensor;

    // set data
    tensor.data = const_cast<byte_t*>(array.get_data());
    tensor.device = get_dlpack_device(array);
    tensor.ndim = std::int32_t(2);
    tensor.dtype = convert_dal_to_dlpack_type(dtype);

    // set shape int64_t, which is the output type of a homogen table and for shape and strides
    if (layout == dal::data_layout::row_major) {
        tensor.shape =
            new std::int64_t[4]{ row_count, column_count, column_count, std::int64_t(1) };
    }
    else {
        tensor.shape = new std::int64_t[4]{ row_count, column_count, std::int64_t(1), row_count };
    }

    // take strategy from dpctl tensors in having a single array allocation by tensor.shape.
    tensor.strides = &tensor.shape[2];
    tensor.byte_offset = std::uint64_t(0);

    return tensor;
}

static void free_capsule(PyObject* cap) {
    DLManagedTensor* dlm = nullptr;
    if (PyCapsule_IsValid(cap, "dltensor")) {
        dlm = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(cap, "dltensor"));
        if (dlm->deleter) {
            dlm->deleter(dlm);
        }
    }
}

py::capsule construct_dlpack(const dal::table& input) {
    // DLManagedTensor is used instead of DLManagedTensorVersioned
    // due to major frameworks not yet supporting the latter.
    DLManagedTensor* dlm = new DLManagedTensor;

    // check table type and expose oneDAL array
    if (input.get_kind() != dal::homogen_table::kind())
        throw pybind11::type_error("Unsupported table type for dlpack conversion");

    auto homogen_input = reinterpret_cast<const dal::homogen_table&>(input);
    dal::array<byte_t> array = dal::detail::get_original_data(homogen_input);
    dlm->manager_ctx = static_cast<void*>(new dal::array<byte_t>(array));

    // set tensor
    dlm->dl_tensor = construct_dlpack_tensor(array,
                                             homogen_input.get_row_count(),
                                             homogen_input.get_column_count(),
                                             homogen_input.get_metadata().get_data_type(0),
                                             homogen_input.get_data_layout());

    // generate tensor deleter
    dlm->deleter = [](struct DLManagedTensor* self) -> void {
        auto stored_array = static_cast<dal::array<byte_t>*>(self->manager_ctx);
        if (stored_array) {
            delete stored_array;
        }
        delete[] self->dl_tensor.shape;
        delete self;
    };

    // create capsule
    py::capsule capsule(static_cast<void*>(dlm), "dltensor", free_capsule);
    return capsule;
}

py::object dlpack_memory_order(py::object obj) {
    DLManagedTensor* dlm;
    DLManagedTensorVersioned* dlmv;
    DLTensor tensor;
    bool versioned;

    // extract __dlpack__ attribute from the input obj. This function should
    // only be called if the attribute has been checked.
    py::capsule caps = obj.attr("__dlpack__")();

    tensor = get_dlpack_tensor(caps, dlm, dlmv, versioned);

    switch (get_dlpack_layout(tensor)) {
        case dal::data_layout::row_major: return py::str("C"); break;
        case dal::data_layout::column_major: return py::str("F"); break;
        default: return py::none();
    }
};

} // namespace oneapi::dal::python::dlpack

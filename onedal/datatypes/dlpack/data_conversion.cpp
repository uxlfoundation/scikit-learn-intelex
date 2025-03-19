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
    bool versioned = false;
    DLManagedTensor* dlm;
    DLManagedTensorVersioned* dlmv;
    DLTensor tensor;
    dal::data_type dtype;

    // extract __dlpack__ attribute from the input obj. This function should
    // only be called if the attribute has been checked.
    py::capsule caps = obj.attr("__dlpack__")();

    // two different types of dlpack managed tensors are possible, with
    // DLManagedTensor likely to be removed from future versions of dlpack.
    // collect important aspects necessary for the macro offloading.
    PyObject* capsule = caps.ptr();
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        dlm = caps.get_pointer<DLManagedTensor>();
        tensor = dlm->dl_tensor;
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        dlmv = caps.get_pointer<DLManagedTensorVersioned>();
        if (dlmv->version.major > DLPACK_MAJOR_VERSION) {
            throw std::runtime_error("dlpack tensor version newer than supported");
        }
        tensor = dlmv->dl_tensor;
        versioned = true;
    }
    else {
        throw std::runtime_error("unable to extract dltensor");
    }

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
} // namespace oneapi::dal::python::dlpack

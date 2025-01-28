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

#include <type_traits>

#include "oneapi/dal/table/homogen.hpp"
#include "oneapi/dal/table/detail/homogen_utils.hpp"

#include "onedal/datatypes/dlpack/data_conversion.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "onedal/common/sycl_interfaces.hpp"
#endif // ONEDAL_DATA_PARALLEL

using namespace pybind11::literals;

namespace oneapi::dal::python::dlpack {

template <typename T, typename managed_t>
inline dal::homogen_table convert_to_homogen_impl(py::object obj,   managed_t& dlm_tensor, py::object q_obj) {
    dal::homogen_table res{};
    DLTensor tensor = dlm_tensor.dl_tensor;
    // Versioned has a readonly flag that can be used to block modification
    bool readonly = std::is_same_v<managed_t, DLManagedTensorVersioned> &&
                    (dlm_tensor.flags & DLPACK_FLAG_BITMASK_READ_ONLY != 0);

    // generate queue from dlpack device information
#ifdef ONEDAL_DATA_PARALLEL
    sycl::queue queue;

    if (tensor.device.device_type == DLDeviceType::kDLOneAPI) {
        queue = q_obj != py::none() ? get_queue_from_python(q_obj)
                                    : get_queue_by_device_id(tensor.device.device_id);
    }
    else if (tensor.device.device_type != DLDeviceType::kDLCPU) {
        throw std::runtime_error("Input array not located on a supported device or CPU");
    }
#else
    // check device, if device is not a oneAPI device or cpu, throw an error. There is no standardized way to move data
    // across devices (even though the 'to_device' function exists in the array_api standard) as devices are not standardized.
    // They are standardized in dlpack, but conversion must be done at a higher level in a case-by-case basis and is therefore
    // outside the scope of to_table. If it is a oneAPI device but sklearnex is not using the dpc backend throw a special error.
    if (tensor.device.device_type == DLDeviceType::kDLOneAPI) {
        throw std::runtime_error(
            "Input array located on a oneAPI device, but sklearnex installation does not have SYCL support.");
    }
    else if (tensor.device.device_type != DLDeviceType::kDLCPU) {
        throw std::runtime_error("Input array not located on CPU");
    }

#endif // ONEDAL_DATA_PARALLEL

    // get and check dimensions
    const std::int32_t ndim = get_ndim(tensor);

    // get shape, if 1 dimensional, force col count to 1
    std::int64_t row_count, col_count;
    row_count = tensor.shape[0];
    col_count = ndim == 1 ? 1l : tensor.shape[1];

    // get data layout for homogeneous check
    const dal::data_layout layout = get_dlpack_layout(tensor, row_count, col_count);

    // unusual data format found, try to make contiguous, otherwise throw error
    if (layout == dal::data_layout::unknown) {
        // NOTE: this will make a C-contiguous deep copy of the data
        // if possible, this is expected to be a special case
        py::object copy;
        if (py::hasattr(obj, "copy")) {
            copy = obj.attr("copy")();
        }
        else if (py::hasattr(obj, "__array_namespace__")) {
            const auto space = obj.attr("__array_namespace__")();
            copy = space.attr("asarray")(obj, "copy"_a = true);
        }
        else {
            throw std::runtime_error("Wrong strides");
        }
        res = convert_to_homogen_impl<T, managed_t>(copy, dlm_tensor, q_obj);
        return res;
    }

    // Get pointer to the data following dlpack.h conventions.
    const auto* const ptr = reinterpret_cast<const T*>(tensor.data); //+ tensor.byte_offset);
    
    // if a nullptr, return an empty.
    if (!ptr)
    {
        return res;
    }

    // create the dlpack deleter, which requires calling the deleter in the dlpackmanagedtensor
    // and decreasing the object's reference count
    const auto deleter = [dlm_tensor](const T *data) {
        if (dlm_tensor.deleter != nullptr) {
            dlm_tensor.deleter(&dlm_tensor);
        }
    };

#ifdef ONEDAL_DATA_PARALLEL
    if (tensor.device.device_type == DLDeviceType::kDLOneAPI) {
        if (readonly) {
            res = dal::homogen_table(queue,
                                     ptr,
                                     row_count,
                                     col_count,
                                     deleter,
                                     std::vector<sycl::event>{},
                                     layout);
        }
        else {
            auto* const mut_ptr = const_cast<T*>(ptr);
            res = dal::homogen_table(queue,
                                     mut_ptr,
                                     row_count,
                                     col_count,
                                     deleter,
                                     std::vector<sycl::event>{},
                                     layout);
        }
        obj.inc_ref();
        return res;
    }
#endif

    if (readonly) {
        res = dal::homogen_table(ptr, row_count, col_count, deleter, layout);
    }
    else {
        auto* const mut_ptr = const_cast<T*>(ptr);
        res = dal::homogen_table(mut_ptr, row_count, col_count, deleter, layout);
    }
    // Towards the python object memory model increment the python object reference
    // count due to new reference by oneDAL table pointing to that object.
    obj.inc_ref();
    return res;
}

dal::table convert_to_table(py::object obj, py::object q_obj) {
    dal::table res;
    bool versioned = false;
    DLManagedTensor dlm;
    DLManagedTensorVersioned dlmv;
    dal::data_type dtype;
    // extract __dlpack__ attribute from the inp_obj
    // this function should only be called if already checked to have this attr
    // Extract and convert a DLpack data type into a oneDAL dtype.
    py::capsule caps = obj.attr("__dlpack__");

    PyObject* capsule = caps.ptr();
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        dlm = *caps.get_pointer<DLManagedTensor>();
        dtype = convert_dlpack_to_dal_type(dlm.dl_tensor.dtype);
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        dlmv = *caps.get_pointer<DLManagedTensorVersioned>();
        if (dlmv.version.major > DLPACK_MAJOR_VERSION) {
            throw std::runtime_error("dlpack tensor version newer than supported");
        }
        versioned = true;
        dtype = convert_dlpack_to_dal_type(dlmv.dl_tensor.dtype);
    }
    else {
        throw std::runtime_error("unable to extract dltensor");
    }

    // if there is a queue, check that the data matches the necessary precision.
#ifdef ONEDAL_DATA_PARALLEL
    if (!q_obj.is(py::none()) && !q_obj.attr("sycl_device").attr("has_aspect_fp64").cast<bool>() &&
        dtype == dal::data_type::float64) {
        // If the queue exists, doesn't have the fp64 aspect, and the data is float64
        // then cast it to float32
        if (hasattr(obj, "__array_namespace__")) {
            PyErr_WarnEx(
                PyExc_RuntimeWarning,
                "Data will be converted into float32 from float64 because device does not support it",
                1);
            const auto space = obj.attr("__array_namespace__")();
            obj = space.attr("astype")(obj, space.attr("float32"));
            res = convert_to_table(obj, queue);
            return res;
        }
        else {
            throw std::runtime_error("Data has higher precision than the supported device");
        }
    }
#endif // ONEDAL_DATA_PARALLEL

    if (versioned) {
#define MAKE_HOMOGEN_TABLE(CType) res = convert_to_homogen_impl<CType, DLManagedTensorVersioned>(obj, dlmv, q_obj);
        SET_CTYPE_FROM_DAL_TYPE(dtype,
                                MAKE_HOMOGEN_TABLE,
                                throw std::invalid_argument("Found unsupported array type"));
#undef MAKE_HOMOGEN_TABLE
    }
    else {
#define MAKE_HOMOGEN_TABLE(CType) \
    res = convert_to_homogen_impl<CType, DLManagedTensor>(obj, dlm, q_obj);
        SET_CTYPE_FROM_DAL_TYPE(dtype,
                                MAKE_HOMOGEN_TABLE,
                                throw std::invalid_argument("Found unsupported array type"));
#undef MAKE_HOMOGEN_TABLE
    }

    // take ownership of the capsule
    dlpack_take_ownership(caps);
    return res;
}
} // namespace oneapi::dal::python::dlpack

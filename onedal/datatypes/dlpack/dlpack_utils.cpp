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

#include <pybind11/pybind11.h>

#include "oneapi/dal/common.hpp"
#include "oneapi/dal/detail/common.hpp"

#include "onedal/datatypes/dlpack/dlpack.h"
#include "onedal/datatypes/dlpack/dlpack_utils.hpp"
#include "onedal/datatypes/dlpack/dtype_conversion.hpp"

namespace py = pybind11;

namespace oneapi::dal::python::dlpack {

DLTensor get_dlpack_tensor(const py::capsule& caps, bool& readonly) {
    // retrieves the dlpack tensor and sets bool for if it is readonly
    PyObject* capsule = caps.ptr();
    DLTensor tensor;
    if(PyCapsule_IsValid(capsule, "dltensor")){
        const DLManagedTensor& ref = *capsule.get_pointer<DLManagedTensor>();
        tensor = ref.dl_tensor;
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")){
        const DLManagedTensorVersioned& ref = *capsule.get_pointer<DLManagedTensorVersioned>();
        if (ref.version.major > DLPACK_MAJOR_VERSION){
            throw std::runtime_error("dlpack tensor version newer than supported")

        }
        // if the value is non-zero cast it to a bool true
        readonly = reinterpret_cast<bool>(ref.flags & DLPACK_FLAG_BITMASK_READ_ONLY)
        tensor = ref.dl_tensor;
    }
    else{
        throw std::runtime_error("unable to extract dltensor")
    }
    return tensor
}

dal::data_type get_dlpack_dtype_from_capsule(const py::capsule& caps) {
    bool throwaway;
    DLTensor tensor = get_dlpack_tensor(caps, throwaway);
    auto dtype = tensor.dtype;
    return convert_dlpack_to_dal_type(std::move(dtype));
}


void dlpack_take_ownership(const py::capsule& caps) {
    // retrieves the dlpack tensor and sets bool for if it is readonly
    PyObject* capsule = caps.ptr();
    if(PyCapsule_IsValid(capsule, "dltensor")){
        caps.set_name("used_dltensor")        
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")){
        caps.set_name("used_dltensor_versioned")
    }
    else{
        throw std::runtime_error("unable to extract dltensor")
    }
}



inline std::int32_t get_ndim(const DLTensor& tensor) {
    const std::int32_t ndim = tensor.ndim;
    if (ndim > 2)
    {
        throw std::runtime_error("Input array has wrong dimensionality (must be 2d).");
    }
    return ndim;
}

dal::data_layout get_dlpack_layout(const DLTensor& tensor,
                                   const std::int64_t& r_count,
                                   const std::int64_t& c_count) {
    const std:int64_t *strides = tensor.strides;
    // if NULL then row major contiguous (see dlpack.h)
    // if 1 column array, also row major
    // if strides of rows = 1 element, and columns = c_count, also row major
    if (strides == NULL || c_count == 1 || (strides[0] == 1 && strides[1] == c_count)){
        return dal::data_layout::row_major;
    }
    else if (strides[0] == r_count && strides[1] == 1){
        return dal::data_layout::col_major;
    }
    else{
        return dal::data_layout::unknown;
    }

}

bool check_dlpack(tensor_t& tensor) {
    const bool is_nullptr = tensor.data == nullptr;
    return !is_nullptr && (0l <= get_dim_count(tensor));
}

bool check_dlpack(managed_t& managed) {
    return check_dlpack(managed.dl_tensor);
}

bool check_external(const py::capsule& caps) {
    const char* const name = caps.name();
    constexpr const char dltensor[] = "dltensor";
    constexpr std::size_t dltensor_size = sizeof(dltensor);
    constexpr const char used_dltensor[] = "used_dltensor";
    constexpr std::size_t used_dltensor_size = sizeof(used_dltensor);

    const bool is_nullptr = caps.get_pointer<managed_t>() == nullptr;
    const bool is_dltensor = std::strncmp(name, dltensor, dltensor_size) == 0;
    const bool is_used_dltensor = std::strncmp(name, used_dltensor, used_dltensor_size) == 0;

    return !is_nullptr && (is_dltensor || is_used_dltensor);
}

bool check_dlpack(const py::capsule& caps) {
    const bool external = check_external(caps);
    return external && check_dlpack(get_managed(caps));
}

void assert_dlpack(const py::capsule& caps) {
    constexpr const char msg[] = "Ill-formed \"dltensor\"";
    if (!check_dlpack(caps)) {
        throw std::runtime_error(msg);
    }
}


std::int64_t get_stride_by_dim(std::int64_t dim, tensor_t& tensor) {
    constexpr const char err[] = "Out of the number of dimensions";
    if (tensor.strides == nullptr) {
        throw std::runtime_error("Not implemented yet");
    }
    else {
        if (dim < 0l || get_dim_count(tensor) <= dim) {
            throw std::out_of_range(err);
        }
        return tensor.strides[dim];
    }
}

std::int64_t get_stride_by_dim(std::int64_t dim, managed_t& managed) {
    return get_stride_by_dim(dim, managed.dl_tensor);
}

std::int64_t get_stride_by_dim(std::int64_t dim, const py::capsule& caps) {
    assert_dlpack(caps);
    return get_stride_by_dim(dim, get_managed(caps));
}

void assert_pointer(dal::data_type dt, void* ptr, const py::capsule& caps) {
    if (get_dtype(caps) != dt) {
        throw std::runtime_error( //
            "Attempting to deleter data of another type");
    }
    if (get_raw_ptr(caps) != ptr) {
        throw std::runtime_error( //
            "Attempting to access wrong data");
    }
}

std::ptrdiff_t get_offset(tensor_t& tensor) {
    const std::uint64_t raw = tensor.byte_offset;
    return detail::integral_cast<std::ptrdiff_t>(raw);
}

void* get_raw_ptr(tensor_t& tensor) {
    auto* ptr = tensor.data;
    auto raw = reinterpret_cast<std::uintptr_t>(ptr);
    auto res = raw + get_offset(tensor);
    return reinterpret_cast<void*>(res);
}

void* get_raw_ptr(managed_t& managed) {
    return get_raw_ptr(managed.dl_tensor);
}

void* get_raw_ptr(const py::capsule& caps) {
    assert_dlpack(caps);
    return get_raw_ptr(get_managed(caps));
}

// Kinda replicates logic from here:
// https://dmlc.github.io/dlpack/latest/python_spec.html
void delete_dlpack(DLManagedTensor* const managed) {
    if (managed->deleter != nullptr) {
        managed->deleter(managed);
    }
}

void delete_dlpack(const py::capsule& caps) {
    assert_dlpack(caps);
    DLManagedTensor* const ptr = //
        caps.get_pointer<DLManagedTensor>();
    delete_dlpack(ptr);
}

} // namespace oneapi::dal::python::dlpack

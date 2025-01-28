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

void dlpack_take_ownership(py::capsule& caps) {
    // retrieves the dlpack tensor and sets bool for if it is readonly
    PyObject* capsule = caps.ptr();
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        caps.set_name("used_dltensor");
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")) {
        caps.set_name("used_dltensor_versioned");
    }
    else {
        throw std::runtime_error("unable to extract dltensor");
    }
}

inline std::int32_t get_ndim(const DLTensor& tensor) {
    const std::int32_t ndim = tensor.ndim;
    if (ndim > 2) {
        throw std::runtime_error("Input array has wrong dimensionality (must be 2d).");
    }
    return ndim;
}

dal::data_layout get_dlpack_layout(const DLTensor& tensor) {
    // get shape, if 1 dimensional, force col count to 1
    std::int64_t row_count, col_count;
    col_count = get_ndim(tensor) == 1 ? 1l : tensor.shape[1];
    row_count = tensor.shape[0];
    const std::int64_t* strides = tensor.strides;
    // if NULL then row major contiguous (see dlpack.h)
    // if 1 column array, also row major
    // if strides of rows = 1 element, and columns = c_count, also row major
    if (strides == NULL || c_count == 1 || (strides[0] == 1 && strides[1] == c_count)) {
        return dal::data_layout::row_major;
    }
    else if (strides[0] == r_count && strides[1] == 1) {
        return dal::data_layout::column_major;
    }
    else {
        return dal::data_layout::unknown;
    }
}

bool check_dlpack_oneAPI_device(const DLDeviceType& device) {
    if (device == DLDeviceType::kDLOneAPI) {
#if ONEDAL_DATA_PARALLEL
        return true;
#else
        throw std::runtime_error(
            "Input array located on a oneAPI device, but sklearnex installation does not have SYCL support.");
#endif // ONEDAL_DATA_PARALLEL
    }
    else if (device != DLDeviceType::kDLCPU) {
#if ONEDAL_DATA_PARALLEL
        throw std::runtime_error("Input array not located on a supported device or CPU");
#else
        throw std::runtime_error("Input array not located on CPU");
#endif // ONEDAL_DATA_PARALLEL
    }
    return false;
}

inline py::object regenerate_layout(const py::object& obj) {
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
    return copy;
}

inline py::object reduce_precision(const py::object& obj) {
    py::object copy;

    // If the queue exists, doesn't have the fp64 aspect, and the data is float64
    // then cast it to float32
    if (hasattr(obj, "__array_namespace__")) {
        PyErr_WarnEx(
            PyExc_RuntimeWarning,
            "Data will be converted into float32 from float64 because device does not support it",
            1);
        const auto space = obj.attr("__array_namespace__")();
        copy = space.attr("astype")(obj, space.attr("float32"));
    }
    else {
        throw std::runtime_error("Data has higher precision than the supported device");
    }
    return copy;
}

} // namespace oneapi::dal::python::dlpack

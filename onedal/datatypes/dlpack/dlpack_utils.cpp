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

using namespace pybind11::literals;

void dlpack_take_ownership(py::capsule& caps) {
    // take a dlpack tensor and claim ownership by changing its name
    // this will block the destructor of the managed dlpack tensor
    // unless the dal table calls it.
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

std::int32_t get_ndim(const DLTensor& tensor) {
    // check if > 2 dimensional, and return the number of dimensions
    const std::int32_t ndim = tensor.ndim;
    if (ndim != 2 && ndim != 1) {
        throw std::length_error("Input array has wrong dimensionality (must be 2d).");
    }
    return ndim;
}

dal::data_layout get_dlpack_layout(const DLTensor& tensor) {
    // determine the layout of a dlpack tensor
    // get shape, if 1 dimensional, force col count to 1
    const std::int64_t r_count = tensor.shape[0];
    const std::int64_t c_count = get_ndim(tensor) > 1 ? tensor.shape[1] : 1l;

    const std::int64_t* strides = tensor.strides;

    // if NULL then row major contiguous (see dlpack.h)
    // if 1 column array, also row major
    // if strides of rows = c_count elements, and columns = 1, also row major
    if (strides == NULL || c_count == 1 || (strides[0] == c_count && strides[1] == 1)) {
        return dal::data_layout::row_major;
    }
    else if (strides[0] == 1 && strides[1] == r_count) {
        return dal::data_layout::column_major;
    }
    else {
        return dal::data_layout::unknown;
    }
}

bool check_dlpack_oneAPI_device(const DLDeviceType& device) {
    // check if dlpack tensor is 1) supported and 2) on a SYCL device

    if (device == DLDeviceType::kDLOneAPI) {
        return true;
    }
    else if (device != DLDeviceType::kDLCPU) {
#if ONEDAL_DATA_PARALLEL
        throw std::invalid_argument("Input array not located on a supported device or CPU");
#else
        throw std::invalid_argument("Input array not located on CPU");
#endif
    }
    return false;
}

py::object regenerate_layout(const py::object& obj) {
    // attempt to use native python commands to get a C- or F-contiguous array
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

py::object reduce_precision(const py::object& obj) {
    // attempt to use native python commands to down convert fp64 data to fp32
    py::object copy;
    if (hasattr(obj, "__array_namespace__")) {
        PyErr_WarnEx(
            PyExc_RuntimeWarning,
            "Data will be converted into float32 from float64 because device does not support it",
            1);
        const auto space = obj.attr("__array_namespace__")();
        copy = space.attr("astype")(obj, space.attr("float32"));
    }
    else {
        throw std::invalid_argument("Data has higher precision than supported by the device");
    }
    return copy;
}

DLTensor get_dlpack_tensor(const py::capsule& caps,
                           DLManagedTensor*& dlm,
                           DLManagedTensorVersioned*& dlmv,
                           bool& versioned) {
    // two different types of dlpack managed tensors are possible, with
    // DLManagedTensor likely to be removed from future versions of dlpack.
    // collect important aspects necessary for use in conversion.
    DLTensor tensor;

    PyObject* capsule = caps.ptr();
    if (PyCapsule_IsValid(capsule, "dltensor")) {
        dlm = caps.get_pointer<DLManagedTensor>();
        tensor = dlm->dl_tensor;
        versioned = false;
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
    return tensor;
}
} // namespace oneapi::dal::python::dlpack

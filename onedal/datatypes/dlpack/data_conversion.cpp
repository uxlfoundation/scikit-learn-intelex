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
#include "onedal/datatypes/dlpack/data_conversion.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "onedal/common/sycl_interfaces.hpp"
#endif // ONEDAL_DATA_PARALLEL

using namespace pybind11::literals;

namespace oneapi::dal::python::dlpack {

template <typename T, typename managed_t>
inline dal::homogen_table convert_to_homogen_impl(const managed_t& dlm_tensor, py::object q_obj) { 
    dal::table res{};
    DLTensor tensor = dlm_tensor.tensor;
    // Versioned has a readonly flag that can be used to block modification
    bool readonly = std::is_same_v< managed_t, DLManagedTensorVersioned>::value && reinterpret_cast<bool>(dlm_tensor.flags & DLPACK_FLAG_BITMASK_READ_ONLY);

    // generate queue from dlpack device information
#ifdef ONEDAL_DATA_PARALLEL
    // if built with dpc backend check the queue against the dlpack device and if the same device use the queue instead
    // of generating a new one, otherwise throw an error. If data is on the cpu, modify datatype to match device precision.
    // It must be emphasized that external functionality (i.e. python-side) is required to deliver the queue to to_table.

    if( tensor.device.device_type == DLDeviceType::kDLOneAPI ){

        if (q_obj != py::none()){


        }
        else {
            sycl::queue queue = get_queue_by_device_id(tensor.device.device_id);
        }

    }
    else if (tensor.device.device_type != DLDeviceType::kDLCPU){
        throw std::runtime_error("Input array not located on a supported device or CPU");
    }


#else
    // check device, if device is not a oneAPI device or cpu, throw an error. There is no standardized way to move data
    // across devices (even though the 'to_device' function exists in the array_api standard) as devices are not standardized.
    // They are standardized in dlpack, but conversion must be done at a higher level in a case-by-case basis and is therefore
    // outside the scope of to_table. If it is a oneAPI device but sklearnex is not using the dpc backend throw a special error.
    if( tensor.device.device_type == DLDeviceType::kDLOneAPI ){
        throw std::runtime_error("Input array located on a oneAPI device, but sklearnex installation does not have SYCL support.");
    }
    else if (tensor.device.device_type != DLDeviceType::kDLCPU){
        throw std::runtime_error("Input array not located on CPU");
    }

#endif // ONEDAL_DATA_PARALLEL




    // get and check dimensions
    const std::int32_t ndim = get_ndim(tensor);

    // get shape, if 1 dimensional, force col count to 1
    std::int64_t row_count, col_count;
    row_count = shape[0];
    col_count = ndim == 1 ? 1l: shape[1];

    // get data layout for homogeneous check
    const dal::data_layout layout = get_dlpack_layout(tensor, row_count, col_count);

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
        res = convert_to_table(copy);
        copy.dec_ref();
        return res;
    }

   // Get pointer to the data following dlpack.h conventions.
    const auto* const ptr = reinterpret_cast<const T*>(tensor.data);

    // create the dlpack deleter, which requires calling the deleter in the dlpackmanagedtensor
    // and decreasing the object's reference count


#ifdef ONEDAL_DATA_PARALLEL
    if( queue ){
        if (readonly){
            res = dal::homogen_table(queue,
                                     ptr,
                                     row_count,
                                     col_count,
                                     deleter,
                                     std::vector<sycl::event>{},
                                     layout);

        }
        else {
              auto* const mut_ptr = const_cast<Type*>(ptr);
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

    if (readonly){
    res = dal::homogen_table(ptr,
                             row_count,
                             column_count,
                             deleter,
                             layout);
    }
    else{
        auto* const mut_ptr = const_cast<Type*>(ptr);
        res = dal::homogen_table(mut_ptr,
                                row_count,
                                column_count,
                                deleter,
                                layout);

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
    // extract __dlpack__ attribute from the inp_obj
    // this function should only be called if already checked to have this attr
    // Extract and convert a DLpack data type into a oneDAL dtype.
    py::capsule caps = obj.attr("__dlpack__");

    PyObject* capsule = caps.ptr();
    if(PyCapsule_IsValid(capsule, "dltensor")){
        dlm = *capsule.get_pointer<DLManagedTensor>();
    }
    else if (PyCapsule_IsValid(capsule, "dltensor_versioned")){
        dlmv = *capsule.get_pointer<DLManagedTensorVersioned>();
        if (dmlv.version.major > DLPACK_MAJOR_VERSION){
            throw std::runtime_error("dlpack tensor version newer than supported")

        }
        versioned = true;
    }
    else{
        throw std::runtime_error("unable to extract dltensor")
    }

    const auto dtype = get_dlpack_dtype_from_capsule(caps);

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
        else{
            throw std::runtime_error("Data has higher precision than the supported device");
        }
    }
#endif // ONEDAL_DATA_PARALLEL

    if(versioned)
    {
#define MAKE_HOMOGEN_TABLE(CType) res = convert_to_homogen_impl<CType, DLManagedTensor>(dml, q_obj);
        SET_CTYPE_FROM_DAL_TYPE(dtype,
                        MAKE_HOMOGEN_TABLE,
                        throw std::invalid_argument("Found unsupported array type"));
#undef MAKE_HOMOGEN_TABLE
    } else 
    {
#define MAKE_HOMOGEN_TABLE(CType) res = convert_to_homogen_impl<CType, DLManagedTensorVersioned>(dmlv, q_obj);
        SET_CTYPE_FROM_DAL_TYPE(dtype,
                        MAKE_HOMOGEN_TABLE,
                        throw std::invalid_argument("Found unsupported array type"));
#undef MAKE_HOMOGEN_TABLE

    }

    // take ownership of the capsule
    dlpack_take_ownership(caps);
    return res;

}



template <std::int64_t dim, typename Deleter>
inline std::shared_ptr<dlpack_interface<dim>> convert(const DLManagedTensor& managed,
                                                      Deleter&& deleter) {

    // Get `DLTensor` struct.
    const DLTensor& tensor = managed.dl_tensor;

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

} // namespace oneapi::dal::python::dlpack

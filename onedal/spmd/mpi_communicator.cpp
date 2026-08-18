/*******************************************************************************
* Copyright 2025 Intel Corporation
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

// Standalone MPI communicator factory for the oneDAL SPMD backend.
//
// This is the ONLY translation unit that includes <mpi.h> and links against an
// MPI library. It is built as a separate pybind11 module (``_onedal_spmd_mpi``)
// so that the SPMD algorithm module (``_onedal_py_spmd_dpc``) carries no MPI
// dependency. It creates communicator objects and hands them to the SPMD policy
// constructors exposed by that algorithm module.
//
// The communicator types are registered here (exactly once across all loaded
// modules); the algorithm module merely names them in its policy constructor
// signatures and relies on pybind11's cross-module type sharing (both modules
// are imported by ``onedal/__init__.py`` before any policy is constructed).
//
// A future oneCCL backend (``_onedal_spmd_ccl``) should mirror this file,
// swapping ``spmd::backend::mpi`` for ``spmd::backend::ccl`` and registering the
// same communicator types under a distinct module name.

#include <pybind11/pybind11.h>

#include "oneapi/dal/spmd/communicator.hpp"
#include "oneapi/dal/spmd/mpi/communicator.hpp"

#ifdef ONEDAL_DATA_PARALLEL
#include "onedal/common/sycl_interfaces.hpp"
#endif // ONEDAL_DATA_PARALLEL

namespace py = pybind11;
namespace spmd = oneapi::dal::preview::spmd;

namespace oneapi::dal::python {

using host_comm_t = spmd::communicator<spmd::device_memory_access::none>;
#ifdef ONEDAL_DATA_PARALLEL
using device_comm_t = spmd::communicator<spmd::device_memory_access::usm>;
#endif // ONEDAL_DATA_PARALLEL

// Register the communicator interface shared with the SPMD algorithm module.
// Only the observer methods are exposed; the object is otherwise opaque and is
// meant to be passed straight into a policy constructor.
template <typename Comm>
void instantiate_communicator(py::module& m, const char* name) {
    py::class_<Comm>(m, name)
        .def("get_rank", &Comm::get_rank)
        .def("get_rank_count", &Comm::get_rank_count)
        .def("get_default_root_rank", &Comm::get_default_root_rank)
        .def(
            "is_root_rank",
            [](const Comm& comm, std::int64_t root) {
                return comm.is_root_rank(root);
            },
            py::arg("root") = -1);
}

} // namespace oneapi::dal::python

PYBIND11_MODULE(_onedal_spmd_mpi, m) {
    using namespace oneapi::dal::python;

    m.doc() = "MPI communicator factory for the oneDAL SPMD backend";

    instantiate_communicator<host_comm_t>(m, "communicator_host");
#ifdef ONEDAL_DATA_PARALLEL
    instantiate_communicator<device_comm_t>(m, "communicator_device");
#endif // ONEDAL_DATA_PARALLEL

    // Single factory: with no queue it produces a host (CPU-distributed)
    // communicator; with a SYCL queue it produces a device communicator. An
    // optional MPI_Comm handle (Fortran integer, e.g. from mpi4py's
    // ``MPI.Comm.py2f``) selects a sub-communicator instead of MPI_COMM_WORLD.
    m.def(
        "create_communicator",
        [](const py::object& queue, const py::object& comm_handle) -> py::object {
            using backend_t = spmd::backend::mpi;

            const bool has_handle = !comm_handle.is_none();
            const std::int64_t handle =
                has_handle ? comm_handle.cast<std::int64_t>() : std::int64_t{ -1 };

            if (queue.is_none()) {
                host_comm_t comm = has_handle
                                       ? spmd::make_communicator<backend_t>(handle)
                                       : spmd::make_communicator<backend_t>();
                return py::cast(comm);
            }
#ifdef ONEDAL_DATA_PARALLEL
            sycl::queue q = get_queue_from_python(queue);
            device_comm_t comm = has_handle
                                     ? spmd::make_communicator<backend_t>(q, handle)
                                     : spmd::make_communicator<backend_t>(q);
            return py::cast(comm);
#else
            throw std::invalid_argument(
                "A SYCL queue was provided but _onedal_spmd_mpi was built without "
                "DPC++ support; device communicators are unavailable.");
#endif // ONEDAL_DATA_PARALLEL
        },
        py::arg("queue") = py::none(),
        py::arg("comm_handle") = py::none(),
        "Create an MPI communicator for SPMD execution. Pass a SYCL queue for a "
        "device communicator, or None for a host (CPU) communicator.");

    m.attr("__backend__") = "mpi";
#ifdef ONEDAL_DATA_PARALLEL
    m.attr("__device_support__") = true;
#else
    m.attr("__device_support__") = false;
#endif // ONEDAL_DATA_PARALLEL
}

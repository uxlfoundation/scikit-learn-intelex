/*******************************************************************************
* Copyright 2014 Intel Corporation
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

#include "mpi/mpi_transceiver.h"
#include <Python.h>
#include <condition_variable>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>

static std::shared_ptr<transceiver> s_trsc;
static std::mutex s_mtx;
static std::condition_variable s_cv;
static bool s_initializing = false;

namespace
{
class gil_state_guard
{
public:
    gil_state_guard() : state(PyGILState_Ensure()) {}
    ~gil_state_guard() { PyGILState_Release(state); }
    gil_state_guard(const gil_state_guard &)            = delete;
    gil_state_guard & operator=(const gil_state_guard &) = delete;

private:
    PyGILState_STATE state;
};

struct pyobject_deleter
{
    void operator()(PyObject * obj) const { Py_XDECREF(obj); }
};
using pyobject_ptr = std::unique_ptr<PyObject, pyobject_deleter>;

[[noreturn]] void throw_python_error(const char * message)
{
    if (PyErr_Occurred()) PyErr_Print();
    throw std::runtime_error(message);
}

std::shared_ptr<transceiver> create_transceiver()
{
    gil_state_guard gil;
    const char * modname = std::getenv("D4P_TRANSCEIVER");
    if (!modname) modname = "daal4py.mpi_transceiver";

    pyobject_ptr mod(PyImport_ImportModule(modname));
    if (!mod) throw_python_error("Could not import the daal4py transceiver module");

    pyobject_ptr ptr(PyObject_GetAttrString(mod.get(), "transceiver"));
    if (!ptr) throw_python_error("Transceiver module has no 'transceiver' attribute");

    void * raw = PyLong_AsVoidPtr(ptr.get());
    if (PyErr_Occurred() || !raw)
        throw_python_error("Invalid transceiver pointer exported by Python module");

    auto iface = reinterpret_cast<std::shared_ptr<transceiver_iface> *>(raw);
    return std::make_shared<transceiver>(*iface);
}
} // namespace

std::shared_ptr<transceiver> get_transceiver()
{
    {
        std::unique_lock<std::mutex> lock(s_mtx);
        s_cv.wait(lock, [] { return !s_initializing; });
        if (s_trsc) return s_trsc;
        s_initializing = true;
    }

    std::shared_ptr<transceiver> candidate;
    try
    {
        candidate = create_transceiver();
    }
    catch (...)
    {
        std::lock_guard<std::mutex> lock(s_mtx);
        s_initializing = false;
        s_cv.notify_all();
        throw;
    }

    std::lock_guard<std::mutex> lock(s_mtx);
    s_trsc         = std::move(candidate);
    s_initializing = false;
    s_cv.notify_all();
    return s_trsc;
}

void del_transceiver()
{
    std::shared_ptr<transceiver> old;
    {
        std::unique_lock<std::mutex> lock(s_mtx);
        s_cv.wait(lock, [] { return !s_initializing; });
        old.swap(s_trsc);
    }
    // Destruction can call into MPI and therefore must happen outside the
    // lifecycle mutex. Existing callers retain their own shared ownership.
    old.reset();
}

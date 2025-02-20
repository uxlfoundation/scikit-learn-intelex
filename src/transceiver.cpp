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
#include <mutex>
#include <Python.h>
#include <cstdlib>

// shared pointer, will GC transceiver when shutting down
static std::shared_ptr<transceiver> s_trsc;

static std::mutex s_mtx;

// We thraed-protect a static variable, which is our transceiver object.
// If unset, we get the mutex and initialize.
// We'll initialize at most once, and return the raw pointer to the tranceiver.
// We load a python module to get the actual transceiver implementation.
// We inspect D4P_TRANSCEIVER env var for using a non-default module.
// We throw an exception if something goes wrong (like the module cannot be loaded).
#define CHECK()                                   \
    if (PyErr_Occurred())                         \
    {                                             \
        PyErr_Print();                            \
        PyGILState_Release(gilstate);             \
        throw std::runtime_error("Python Error"); \
    }
transceiver * get_transceiver()
{
    if (!s_trsc)
    {
        std::lock_guard<std::mutex> lock(s_mtx);
        if (!s_trsc)
        {
            auto gilstate = PyGILState_Ensure();

            const char * modname = std::getenv("D4P_TRANSCEIVER");
            if (modname == NULL) modname = "daal4py.mpi_transceiver";

            PyObject * mod = PyImport_ImportModule(modname);
            CHECK();
            PyObject * ptr = PyObject_GetAttrString(mod, "transceiver");
            CHECK();
            void * tcvr = PyLong_AsVoidPtr(ptr);
            Py_XDECREF(mod);
            CHECK();
            PyGILState_Release(gilstate);

            // we expect the tcvr to be a pointer to a (static) shared-pointer object.
            s_trsc.reset(new transceiver(*reinterpret_cast<std::shared_ptr<transceiver_iface> *>(tcvr)));
        }
    }
    return s_trsc.get();
}
#undef CHECK

void del_transceiver()
{
    if (s_trsc)
    {
        std::lock_guard<std::mutex> lock(s_mtx);
        if (s_trsc)
        {
            auto gilstate = PyGILState_Ensure();
            s_trsc.reset();
        }
    }
}

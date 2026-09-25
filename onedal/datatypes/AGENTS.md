# AGENTS.md - Data Conversion (onedal/datatypes/)

## Purpose
Converts NumPy, SciPy sparse, SYCL USM (dpnp), and DLPack inputs to oneDAL tables and back, zero-copy where possible.

## Layout
- `_data_conversion.py`: Python entry points (`to_table`, `from_table`)
- `_dlpack.py`: DLPack helpers
- `table.cpp`: pybind11 registration
- `numpy/`, `sycl_usm/`, `dlpack/`: per-format `data_conversion.cpp` plus dtype helpers
- `tests/test_data.py`: round-trip tests across formats and devices

## Rules
Most review findings in this directory are reference leaks and aliasing bugs, not style.

- Never hold a bare owned `PyObject*`. Wrap it in `py::reinterpret_steal<py::object>(...)` as soon as it is created, so every throwing path releases it. Note that `convert_to_numpy_impl`, `PyArray_New`, and `need_mutable_data()` can all throw.
- Only steal references you own. Borrowed references (attributes of user-provided objects, `PyTuple_GetItem`) use `py::reinterpret_borrow`.
- Check `NULL` returns from the C API (`PyTuple_New`, `PyArray_New`) before using the result.
- `dal::array::wrap` on a `const_cast` pointer produces a non-owning array, so `need_mutable_data()` doesn't copy and the result aliases the source buffer. Whatever owns that buffer must outlive every array built from it.
- Don't call `np.asarray()` on device arrays; it forces a host copy and breaks SYCL/array API inputs. Use `from_dlpack` with an explicit device.
- Errors reach the user as Python exceptions. Throw a C++ exception that pybind11 translates; never write to `std::cerr`, which is not Python's error stream.

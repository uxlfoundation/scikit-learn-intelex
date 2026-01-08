# onedal/* - Low-Level C++ Bindings

## Purpose
Pybind11-based C++ bindings providing the bridge between Python and Intel oneDAL C++ library.

## Key Components
- `datatypes/` - Memory management and array conversions (NumPy, SYCL USM, DLPack)
- `common/` - Policy management, device selection, serialization
- Algorithm modules: `cluster/`, `decomposition/`, `linear_model/`, `svm/`, `ensemble/`, `basic_statistics/`, `covariance/`
- `spmd/` - Distributed computing interfaces

## Testing
```bash
pytest --verbose --pyargs onedal
pytest onedal/tests/
pytest onedal/datatypes/tests/
pytest onedal/common/tests/
```

## For GitHub Copilot

See [onedal/AGENTS.md](../onedal/AGENTS.md) for comprehensive information including:
- Backend system architecture (DPC++/Host selection)
- Data conversion methods and zero-copy patterns
- Algorithm structure and implementation patterns
- Memory management strategies

## Related Instructions
- `general.instructions.md` - Repository setup
- `src.instructions.md` - C++/Cython implementation using onedal
- `sklearnex.instructions.md` - High-level layer built on onedal
- `daal4py.instructions.md` - Alternative interface to onedal

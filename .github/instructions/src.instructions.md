# src/* - Core C++/Cython Implementation

## Purpose
Core C++/Cython implementation layer providing the foundation for the entire stack.

## Key Files
- `daal4py.cpp`: Main Cython interface to oneDAL
- `daal4py.h`: C++ headers and type definitions
- `*_builder.pyx`: Model builder implementations (XGBoost, LightGBM conversion)
- `gettree.pyx`: Tree model extraction utilities
- `mpi/`: Distributed computing infrastructure

## Architecture
- **Cython Interface**: `daal4py.cpp` provides Python↔C++ bridge
- **Memory Management**: `npy4daal.h` handles NumPy array conversions
- **Distributed Computing**: MPI-based implementations in `mpi/`
- **Model Builders**: Cython implementations for external model conversion

## Build Process
1. **Code Generation**: Python scripts generate C++ from oneDAL headers
2. **Cython Compilation**: `.pyx` files compiled to C++
3. **C++ Compilation**: Link with oneDAL libraries
4. **Extension Creation**: Python extension modules

## Development Workflow

See [build-config.instructions.md](build-config.instructions.md) for environment setup.

```bash
# Rebuild after C++/Cython changes
python setup.py build_ext --inplace --force
```

## MPI/Distributed Features
- Located in `src/mpi/`
- Requires MPI installation (`MPIROOT` environment variable)
- Enable with `mpi4py` for distributed sklearn operations
- Disable with `NO_DIST=1` if MPI unavailable

## Testing
```bash
# Test distributed features (requires MPI)
mpirun -n 2 python -m pytest tests/test_daal4py_spmd_examples.py

# Test model builders
pytest tests/test_model_builders.py

# Test core functionality
pytest tests/test_daal4py_serialization.py
```

## Development Notes
- No incremental compilation - full rebuild required for changes
- Use `ccache` for faster development builds
- ASan builds supported for debugging (see INSTALL.md)
- C++ code must follow clang-format style

## Related Instructions
- `general.instructions.md` - Repository setup and build requirements
- `build-config.instructions.md` - Build system and compilation details
- `onedal.instructions.md` - Python bindings that src/ implements
- `daal4py.instructions.md` - Higher-level API built on src/
- See `src/AGENTS.md` for detailed implementation guides
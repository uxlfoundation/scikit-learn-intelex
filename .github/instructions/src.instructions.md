# src/* - Core C++/Cython Implementation

## Purpose
Core C++/Cython implementation layer providing the foundation for the entire stack.

## Key Files
- `daal4py.cpp/.h` - Main Cython interface to oneDAL
- `*_builder.pyx` - Model builder implementations (XGBoost, LightGBM, CatBoost conversion)
- `gettree.pyx` - Tree model extraction utilities
- `mpi/` - Distributed computing infrastructure

## Build Process
Rebuild after C++/Cython changes: `python setup.py build_ext --inplace --force`

## Testing
```bash
mpirun -n 2 python -m pytest tests/test_daal4py_spmd_examples.py
pytest tests/test_model_builders.py
pytest tests/test_daal4py_serialization.py
```

## For GitHub Copilot

See [src/AGENTS.md](../src/AGENTS.md) for comprehensive information including:
- C++/Cython architecture and memory management
- GIL protection patterns and thread safety
- MPI communication layer and distributed algorithms
- Model builder interfaces for external frameworks

## Related Instructions
- `general.instructions.md` - Repository setup
- `build-config.instructions.md` - Build system and compilation
- `onedal.instructions.md` - Python bindings that src/ implements
- `daal4py.instructions.md` - Higher-level API built on src/

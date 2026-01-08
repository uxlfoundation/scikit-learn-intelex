# AGENTS.md - Intel Extension for Scikit-learn

## Quick Context
- **Purpose**: Accelerate scikit-learn using Intel oneDAL optimizations
- **License**: Apache 2.0
- **Languages**: Python, C++, Cython
- **Platforms**: CPU (x86_64, ARM), GPU (Intel via SYCL)

## Architecture (4 Layers)
```
User Apps → sklearnex/ → daal4py/ → onedal/ → Intel oneDAL C++
```

**Layer Functions:**
- `sklearnex/`: sklearn API compatibility and patching system
- `daal4py/`: Direct oneDAL access and model builders
- `onedal/`: Pybind11 bindings and memory management
- `src/`: C++/Cython core implementation

## Entry Points by Use Case

**sklearn acceleration** - Global patching or selective imports from sklearnex

**Native oneDAL** - Direct algorithm access via daal4py for maximum performance

**Model conversion** - Convert XGBoost/LightGBM/CatBoost models to oneDAL format for accelerated inference

## Accelerated Algorithms
- **Clustering**: DBSCAN, K-Means
- **Classification**: SVM, RandomForest, LogisticRegression, NaiveBayes
- **Regression**: LinearRegression, Ridge, Lasso, ElasticNet, SVR
- **Decomposition**: PCA, IncrementalPCA
- **Neighbors**: KNeighbors (classification/regression)
- **Preprocessing**: Scalers, normalizers
- **Statistics**: Basic statistics, covariance

## Device Configuration
GPU offloading and device control available through sklearnex config_context for supported algorithms.

## Performance Patterns
- **Memory**: Zero-copy NumPy↔oneDAL, SYCL USM for GPU
- **Parallelism**: Intel TBB threading, MPI distributed (SPMD), SIMD vectorization
- **Fallbacks**: oneDAL → sklearn cascading fallback on unsupported operations
- **Speedups**: 10-100X acceleration over sklearn (varies by algorithm and data characteristics)

## Key Files for AI Agents
- `sklearnex/dispatcher.py`: Patching system and algorithm dispatch
- `sklearnex/_device_offload.py`: Device selection and offloading
- `onedal/__init__.py`: Backend selection (DPC++/Host)
- `daal4py/__init__.py`: Native API entry point
- `src/`: C++/Cython core (distributed computing, memory management)

## Development Setup

### Prerequisites
- **Python**: 3.9+
- **oneDAL**: 2021.1+ (backwards compatible)
- **Dependencies**: Cython, Jinja2, numpy, pybind11, cmake (see dependencies-dev)

### Build Commands
Install dependencies, set DALROOT environment variable, run setup.py develop for development mode.

### Environment Options
- `DALROOT`: Path to oneDAL (required)
- `MPIROOT`: Path to MPI for distributed support
- `NO_DPC`: Disable GPU support
- `NO_DIST`: Disable distributed computing
- `NO_STREAM`: Disable streaming mode

## Testing Strategy
Core test suites cover legacy tests, native oneDAL (daal4py), sklearn compatibility (sklearnex), low-level backend (onedal), and global patching. MPI required for distributed (SPMD) testing.

## Performance Expectations

### Algorithm Support
oneDAL acceleration requires:
- Dense data (not sparse)
- Supported dtypes: float32, float64
- Contiguous memory layout preferred
- Algorithm-specific parameter compatibility

### GPU Support Status
- **Full GPU**: DBSCAN, K-Means, PCA, KNeighbors
- **Limited GPU**: LogisticRegression, SVM
- **CPU Only**: RandomForest, Ridge, IncrementalPCA

### Error Handling
Fallback chain: oneDAL → sklearn → error. Configurable via allow_sklearn_after_onedal setting.

### Memory Requirements
oneDAL requires contiguous data for zero-copy operations. C-contiguous preferred over Fortran-contiguous.

## GPU Hardware
**Supported Intel GPUs**: Integrated (UHD Graphics, Iris Xe), Discrete (Arc series)
**Requirements**: SYCL/DPC++ support, Intel oneAPI toolkit, Unified Shared Memory (USM)

## Version Compatibility
- **Python**: 3.9+
- **scikit-learn**: 1.0+
- **oneDAL**: 2021.1+ (backwards compatible only)

### Version Support Policy

**Python Support**:
The project supports officially maintained Python versions. Support for newly released Python versions may be delayed, and support for older versions may extend beyond official end-of-life to accommodate user needs.

**scikit-learn Support**:
The project aims to support the last 4 scikit-learn releases. sklearn 1.0 is maintained as a special case due to continued usage in production environments.

**Exact Versions**:
For exact dependency versions, always refer to:
- `dependencies-dev` - Build dependencies
- `requirements-test.txt` - Test dependencies
- `setup.py` - Python version classifiers

Documentation may lag behind actual supported versions.

## Code Generation
The generator/ directory contains automated code generation from oneDAL C++ headers to Python bindings. Modify generator/wrappers.py to add new oneDAL algorithms; use direct Python implementation for sklearn compatibility layers.

## Component Documentation
- `sklearnex/AGENTS.md`: API patterns, device offloading
- `daal4py/AGENTS.md`: Native oneDAL bindings, model builders
- `onedal/AGENTS.md`: Pybind11 implementation, memory management
- `src/AGENTS.md`: C++/Cython core, distributed computing
- `examples/AGENTS.md`: Usage patterns and example scripts
- `tests/AGENTS.md`: Testing infrastructure, validation patterns
- `.ci/AGENTS.md`: CI/CD pipeline and build infrastructure
- `doc/AGENTS.md`: Documentation build system

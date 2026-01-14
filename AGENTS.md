# AGENTS.md - Intel Extension for Scikit-learn

## Quick Context
- **Purpose**: Accelerate scikit-learn using Intel oneDAL optimizations
- **License**: Apache 2.0
- **Languages**: Python, C++, Cython
- **Platforms**: CPU (x86_64, ARM), GPU (Intel via SYCL)

## Architecture (4 Layers)
```text
User Apps → sklearnex/ ⇒ {daal4py/, onedal/} → Intel oneDAL C++
                             ↓          ↓
                       (Cython ext)  (pybind11)
```

**Layer Functions:**
- `sklearnex/`: sklearn API compatibility, patching, uses both daal4py and onedal
- `daal4py/`: Direct oneDAL access via Cython, model builders, legacy compatibility
- `onedal/`: Modern pybind11 bindings, memory management, GPU/CPU backend selection
- `src/`: C++/Cython core implementation shared by both bindings

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

## Installation

### Quick Install (Users)
```bash
# Recommended: via conda
conda install -c conda-forge scikit-learn-intelex

# Or via pip
pip install scikit-learn-intelex
```

### From Source (Contributors)
```bash
# 1. Install oneDAL and set DALROOT
export DALROOT=/path/to/onedal  # Required

# 2. Install dependencies
pip install -r requirements-test.txt

# 3. Build in development mode
python setup.py develop
```

### Environment Variables
- `DALROOT`: Path to oneDAL (required for source builds)
- `MPIROOT`: Path to MPI for distributed support
- `NO_DPC`: Disable GPU support
- `NO_DIST`: Disable distributed computing
- `NO_STREAM`: Disable streaming mode

## Testing Strategy
Core test suites cover legacy tests, native oneDAL (daal4py), sklearn compatibility (sklearnex), low-level backend (onedal), and global patching. MPI required for distributed (SPMD) testing.

## Performance Expectations

### Algorithm Support
oneDAL acceleration requires:
- Supported dtypes: float32, float64
- Contiguous memory layout preferred
- Algorithm-specific parameter compatibility

### GPU Support Status
- **Full GPU**: DBSCAN, K-Means, PCA, KNeighbors
- **Limited GPU**: LogisticRegression, SVM
- **CPU Only**:  Ridge, IncrementalPCA

### Error Handling
Fallback chain: oneDAL → sklearn → error. Configurable via allow_sklearn_after_onedal setting.

### Memory Requirements
oneDAL requires contiguous data for zero-copy operations. C-contiguous preferred over Fortran-contiguous.

## GPU Hardware
**Supported Intel GPUs**: Integrated (UHD Graphics, Iris Xe), Discrete (Arc series), Datacenter (Flex)
**Requirements**: SYCL/DPC++ support, Intel oneAPI toolkit, Unified Shared Memory (USM)

### GPU Setup & Troubleshooting
**Verify GPU availability:**
```bash
python -c "import dpctl; print(dpctl.get_devices())"
```

**Common GPU issues:**
- **"No GPU device found"** → Install Intel GPU drivers and `intel-opencl-icd` (Linux) or Intel Graphics drivers (Windows)
- **`ImportError: dpctl`** → Install GPU runtime: `pip install dpctl dpnp`
- **Fallback to CPU** → Check `verbose=True` to see reason (unsupported param, sparse data, etc.)
- **Out of memory** → Reduce data size or use `target_offload="cpu"`

## Common Errors

**Build/Setup:**
- **"Not set DALROOT variable"** → Export DALROOT pointing to oneDAL installation
- **"MPIROOT is not set"** → For distributed mode, set MPIROOT or use `NO_DIST=1`

**Runtime:**
- **"oneDAL backend not available"** → Algorithm/parameter not supported by oneDAL, fallback to sklearn
- **"Unsupported parameter"** → Parameter value incompatible with oneDAL (check documentation)
- **"Sparse data not supported"** → Convert to dense or use sklearn (except SVM, NaiveBayes support CSR)
- **MPI errors in SPMD** → Call `daal4py.daalinit()` before distributed operations, `daalfini()` at end

## Version Compatibility

**For current supported versions, always check:**
- `setup.py` - Python version classifiers
- `requirements-test.txt` - scikit-learn and runtime dependencies
- `dependencies-dev` - Build dependencies

### Version Support Policy

**Python**:
Supports officially maintained Python versions. Support for newly released versions may be delayed; support for older versions may extend beyond EOL to accommodate user needs.

**scikit-learn**:
Aims to support the last 4 scikit-learn releases. sklearn 1.0 maintained as special case for production environments.

**oneDAL**:
Backwards compatible with oneDAL 2021.1+. Forward compatibility not guaranteed.

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

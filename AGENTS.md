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

**Key Layer Functions:**
- `sklearnex/`: sklearn API compatibility + patching
- `daal4py/`: Direct oneDAL access + model builders
- `onedal/`: Pybind11 bindings + memory management
- `src/`: C++/Cython core implementation

## Entry Points by Use Case

**For sklearn acceleration:**
```python
from sklearnex import patch_sklearn; patch_sklearn()
# OR direct import
from sklearnex.cluster import DBSCAN
```

**For native oneDAL performance:**
```python
import daal4py as d4p
algorithm = d4p.dbscan(epsilon=0.5, minObservations=5)
```

**For model conversion:**
```python
from daal4py.mb import convert_model
d4p_model = convert_model(xgb_model)  # XGBoost→oneDAL
```

## Accelerated Algorithms
- **Clustering**: DBSCAN, K-Means
- **Classification**: SVM, RandomForest, LogisticRegression, NaiveBayes
- **Regression**: LinearRegression, Ridge, Lasso, ElasticNet, SVR
- **Decomposition**: PCA, IncrementalPCA
- **Neighbors**: KNeighbors (classification/regression)
- **Preprocessing**: Scalers, normalizers

## Device Configuration
```python
from sklearnex import config_context

# GPU offloading
with config_context(target_offload="gpu:0"):
    model.fit(X, y)

# Force CPU
with config_context(target_offload="cpu"):
    model.fit(X, y)
```

## Performance Patterns
- **Memory**: Zero-copy NumPy↔oneDAL, SYCL USM for GPU
- **Parallelism**: Intel TBB threading, MPI distributed, SIMD vectorization
- **Fallbacks**: oneDAL → sklearn → error cascade

## Key Files for AI Agents
- `sklearnex/dispatcher.py`: Patching system (line 36: `get_patch_map_core`)
- `sklearnex/_device_offload.py`: Device dispatch (line 72: `dispatch`)
- `onedal/__init__.py`: Backend selection
- `daal4py/__init__.py`: Native API entry
- `src/`: C++/Cython core (distributed computing, memory management)

## Development Environment Setup

### Prerequisites
- **Python**: 3.9-3.13 (verified in setup.py classifiers and README.md badges)
- **oneDAL**: 2021.1+ (backwards compatible, verified in INSTALL.md)
- **Dependencies**: Cython==3.1.1, Jinja2==3.1.6, numpy>=2.0.1, pybind11==2.13.6, cmake==4.0.2 (verified in dependencies-dev file)

### Build Commands
```bash
# Development setup
pip install -r dependencies-dev  # Verified: contains Cython, Jinja2, numpy, pybind11, cmake
export DALROOT=/path/to/onedal    # Required (verified in setup.py:53-59)
export MPIROOT=/path/to/mpi       # For distributed support (verified in setup.py:95-100)
python setup.py develop           # Development mode

# Environment options
export NO_DPC=1                 # Disable GPU support
export NO_DIST=1                # Disable distributed computing
export NO_STREAM=1              # Disable streaming mode
```

### Testing Strategy
```bash
# Core test suites (from conda-recipe/run_test.sh)
pytest --verbose -s tests/                    # Legacy tests
pytest --verbose --pyargs daal4py            # Native oneDAL tests
pytest --verbose --pyargs sklearnex          # sklearn compatibility
pytest --verbose --pyargs onedal             # Low-level backend
pytest --verbose .ci/scripts/test_global_patch.py  # Global patching

# Distributed testing (requires MPI)
mpirun -n 4 python tests/helper_mpi_tests.py pytest -k spmd --with-mpi --pyargs sklearnex
```

## Performance Expectations

### Benchmarked Speedups
- **General**: 10-100X acceleration (verified in README.md)
- **Training**: Up to 100x speedup mentioned in README.md
- **Inference**: Significant speedup, model builders claim 10-100x for converted models
- **Range**: 1-3 orders of magnitude improvement depending on algorithm/dataset
- **Note**: Specific 27x/36x figures not found in current codebase, general 10-100X claims verified

### Algorithm Support Decision Matrix

**oneDAL Acceleration Criteria** (verified in sklearnex/cluster/dbscan.py:108-138):
```python
def _onedal_supported(self, method_name, *data):
    # Data requirements (verified in DBSCAN implementation)
    - Dense data only (not sp.issparse(X))
    - Supported dtypes: float32, float64
    - Contiguous memory layout preferred

    # Algorithm-specific constraints (verified in actual code)
    - DBSCAN: algorithm in ["auto", "brute"], metric="euclidean" or "minkowski" with p=2
    - Parameter compatibility checks via PatchingConditionsChain
```

**GPU Support Status** (from sklearnex/AGENTS.md):
- **Full GPU**: DBSCAN, K-Means, PCA, KNeighbors
- **Limited GPU**: LogisticRegression (2024.1+), SVM
- **CPU Only**: RandomForest, Ridge, IncrementalPCA

### Error Handling and Fallback Strategy

**Fallback Chain** (verified in onedal/_config.py:45-50):
```python
# Configuration controls fallback behavior
_default_global_config = {
    "target_offload": "auto",               # Auto device selection
    "allow_fallback_to_host": False,        # GPU → CPU fallback
    "allow_sklearn_after_onedal": True,     # oneDAL → sklearn fallback
    "use_raw_input": False,                 # Raw input usage
}
```

**Fallback Triggers**:
1. **Unsupported data**: Sparse matrices, unsupported dtypes
2. **Unsupported parameters**: Algorithm-specific limitations
3. **Hardware constraints**: GPU memory limits, device unavailability
4. **Runtime errors**: oneDAL computation failures

### Memory Management Patterns

**Critical Requirements** (from sklearnex/utils/validation.py):
```python
# oneDAL requires contiguous data - copying avoided for performance
def _onedal_supported_format(X, xp):
    return is_contiguous(X)  # C-contiguous preferred
```

**Data Layout**:
- **Contiguous arrays**: Required for zero-copy operations
- **Data types**: float32/float64 preferred, automatic conversion when needed
- **Memory layout**: C-contiguous > Fortran-contiguous > non-contiguous

### GPU Hardware Requirements

**Supported Intel GPUs**:
- **Integrated**: Intel UHD Graphics, Intel Iris Xe
- **Discrete**: Intel Arc A370M, Arc B580, Arc series
- **Requirements**: SYCL/DPC++ support, Intel oneAPI toolkit
- **Memory**: Unified Shared Memory (USM) support for zero-copy operations

### Version Compatibility

**Supported Versions** (verified in README.md badges and setup.py):
- **Python**: 3.9, 3.10, 3.11, 3.12, 3.13 (verified in setup.py:609-613)
- **scikit-learn**: 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7 (verified in README.md badge)
- **oneDAL**: 2021.1+ (backwards compatible only, verified in INSTALL.md)

### Code Generation vs Manual Implementation

**When to use generator/** (from INSTALL.md build process):
1. **Automatic**: C++ headers → Python bindings (stage 1 of 4-stage build)
2. **Manual Python**: Direct sklearn interface implementations
3. **Generator changes**: Required for new oneDAL algorithms not yet wrapped
4. **Python changes**: Sufficient for parameter handling, validation, sklearn compatibility

### SPMD (Distributed) Usage Guidelines

**When to use SPMD** (from tests/helper_mpi_tests.py, conda-recipe/run_test.sh):
- **Large datasets**: When single-node memory insufficient
- **Supported algorithms**: DBSCAN, K-Means, PCA, Linear Regression
- **Setup**: Requires MPI (Intel MPI or OpenMPI), mpi4py
- **Testing**: `mpirun -n 4` for validation

**MPI Requirements** (from setup.py):
```python
mpi_root = os.environ.get("MPIROOT", os.environ.get("I_MPI_ROOT"))
# Required unless NO_DIST=1
```

## Component Documentation
- `sklearnex/AGENTS.md`: API patterns, device offloading
- `daal4py/AGENTS.md`: Native oneDAL bindings, model builders
- `onedal/AGENTS.md`: Pybind11 implementation, memory management
- `src/AGENTS.md`: C++/Cython core, distributed computing
- `examples/AGENTS.md`: Usage patterns (113 scripts, 19 notebooks)
- `tests/AGENTS.md`: Testing infrastructure, validation patterns
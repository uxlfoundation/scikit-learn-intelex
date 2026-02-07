# AGENTS.md - sklearnex Package

## Purpose
Primary sklearn-compatible interface with oneDAL acceleration, device offloading, and transparent patching.

## Core Files
- `dispatcher.py`: Patching system and algorithm mapping
- `_config.py`: Configuration context (target_offload, fallback controls)
- `_device_offload.py`: Device dispatch logic
- `base.py`: oneDALEstimator base class

## Usage Patterns

### Global Patching
Patch all sklearn imports to use oneDAL via `patch_sklearn()`. All subsequent sklearn imports use accelerated versions.

### Selective Patching
Patch specific algorithms only by passing list to `patch_sklearn()`.

### Direct Import
Import directly from sklearnex for guaranteed oneDAL acceleration without patching.

### Status Check
Use `sklearn_is_patched()` to verify patching state.

## Configuration API

Use `config_context()` from `sklearnex` to control behavior:

```python
from sklearnex import config_context

with config_context(target_offload="gpu"):
    model.fit(X, y)  # Uses GPU if supported
```

### All Configuration Options

**Device Control:**
- `target_offload`: Device selection
  - `"auto"` (default): Automatic based on data location
  - `"cpu"`: Force CPU execution
  - `"gpu"` or `"gpu:0"`: GPU acceleration (requires Intel GPU + dpctl)
  - `dpctl.SyclQueue` object: Explicit queue control

**Fallback Control:**
- `allow_fallback_to_host`: `bool` (default `True`) - Enable GPU → CPU fallback
- `allow_sklearn_after_onedal`: `bool` (default `True`) - Enable oneDAL → sklearn fallback

**Advanced:**
- `array_api_dispatch`: `bool` - Enable Array API namespace dispatch, should be set to `True` when running GPU algorithms to run data validation without host transfers

## Algorithm Support

### Dispatch Flow
1. Check GPU support → Use GPU oneDAL if available
2. Check CPU support → Use CPU oneDAL if available
3. Fallback → Use original sklearn implementation

### Condition Checking
Algorithms implement `_onedal_cpu_supported()` and `_onedal_gpu_supported()` to validate data types, parameters, and hardware availability via PatchingConditionsChain.

## Supported Algorithms
- **Clustering**: DBSCAN, K-Means
- **Linear Models**: LogisticRegression, Ridge, LinearRegression
- **Ensemble**: RandomForestClassifier, RandomForestRegressor
- **Decomposition**: PCA, IncrementalPCA (preview)
- **Neighbors**: KNeighborsClassifier, KNeighborsRegressor
- **SVM**: SVC, SVR, NuSVC, NuSVR

## GPU Support Status
doc/sources/algorithms.rst details level of support of each algo on CPU, GPU, and multi-GPU spmd configurations

## Distributed Computing (SPMD)
Located in `sklearnex/spmd/`. Same API as standard sklearnex, distributed across MPI nodes. Import from `sklearnex.spmd.*` instead of `sklearnex.*`.

## Preview Features
Experimental algorithms in `sklearnex/preview/`. Activate via `export SKLEARNEX_PREVIEW=1`.

**Available in Preview:**
- `sklearnex.preview.decomposition.IncrementalPCA`: Enhanced incremental PCA
- `sklearnex.preview.covariance`: Enhanced covariance implementations

**Note**: Preview APIs may change without deprecation warnings.

### Fallback Chain
oneDAL GPU → oneDAL CPU → sklearn → Error

### Common Fallback Triggers
- Sparse data (most algorithms require dense)
- Unsupported parameters
- GPU memory limits
- Unsupported data types

## Key Implementation Patterns
- Use config_context for device control and fallback configuration
- Check `_onedal_*_supported()` methods to understand algorithm constraints
- Understand three-tier fallback: GPU → CPU → sklearn
- SPMD variants in `spmd/` subdirectory for distributed execution
- Preview features require explicit environment variable activation
- Direct imports from sklearnex guarantee acceleration without patching

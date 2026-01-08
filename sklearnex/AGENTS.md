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

### Device Control
Control execution device via `config_context(target_offload=...)`:
- `"gpu:0"`: GPU acceleration
- `"cpu"`: Force CPU execution
- `"auto"`: Automatic device selection based on data location (default)

### Fallback Control
Configure fallback behavior:
- `allow_fallback_to_host`: Enable GPU → CPU fallback
- `allow_sklearn_after_onedal`: Enable oneDAL → sklearn fallback

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
- **Decomposition**: PCA, IncrementalPCA
- **Neighbors**: KNeighborsClassifier, KNeighborsRegressor
- **SVM**: SVC, SVR, NuSVC, NuSVR

## GPU Support Status
- **Full GPU**: DBSCAN, K-Means, PCA, KNeighbors
- **Limited GPU**: LogisticRegression, SVM
- **CPU Only**: RandomForest, Ridge, IncrementalPCA

## Distributed Computing (SPMD)
Located in `sklearnex/spmd/`. Same API as standard sklearnex, distributed across MPI nodes. Import from `sklearnex.spmd.*` instead of `sklearnex.*`.

## Preview Features
Experimental algorithms and enhancements in `sklearnex/preview/`. Activate via `export SKLEARNEX_PREVIEW=1`.

## Error Handling

### Fallback Chain
oneDAL GPU → oneDAL CPU → sklearn → Error

### Common Fallback Triggers
- Sparse data (most algorithms require dense)
- Unsupported parameters
- GPU memory limits
- Unsupported data types

## For AI Agents
- Use config_context for device control and fallback configuration
- Check `_onedal_*_supported()` methods to understand algorithm constraints
- Understand three-tier fallback: GPU → CPU → sklearn
- SPMD variants in `spmd/` subdirectory for distributed execution
- Preview features require explicit environment variable activation
- Direct imports from sklearnex guarantee acceleration without patching

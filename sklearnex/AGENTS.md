# AGENTS.md - sklearnex Package

## Purpose
**Primary sklearn-compatible interface** with oneDAL acceleration

## Core Files
- `dispatcher.py`: Patching system (`get_patch_map_core` line 36)
- `_config.py`: Configuration (`target_offload`, `allow_fallback_to_host`)
- `_device_offload.py`: Device dispatch (`dispatch` function line 72)
- `base.py`: oneDALEstimator base class

## Usage Patterns

**Global Patching:**
```python
from sklearnex import patch_sklearn
patch_sklearn()  # All sklearn imports use oneDAL
from sklearn.cluster import DBSCAN  # Now accelerated
```

**Selective Patching:**
```python
patch_sklearn(["DBSCAN", "KMeans"])  # Only specific algorithms
```

**Direct Import:**
```python
from sklearnex.cluster import DBSCAN  # Always accelerated
```

**Status Check:**
```python
from sklearnex import sklearn_is_patched
print(sklearn_is_patched())  # True/False
```

## Configuration API

**Device Control:**
```python
from sklearnex import config_context

# GPU acceleration
with config_context(target_offload="gpu:0"):
    model.fit(X, y)

# Force CPU
with config_context(target_offload="cpu"):
    model.fit(X, y)

# Auto device selection
with config_context(target_offload="auto"):  # Default
    model.fit(X, y)
```

**Fallback Control:**
```python
# Allow CPU fallback when GPU fails
with config_context(allow_fallback_to_host=True):
    model.fit(X_gpu, y_gpu)

# Allow sklearn fallback when oneDAL fails
with config_context(allow_sklearn_after_onedal=True):
    model.fit(X, y)
```

## Algorithm Support Conditions

**Implementation Pattern:**
```python
class Algorithm(BaseAlgorithm, oneDALEstimator, _sklearn_Algorithm):
    def _onedal_cpu_supported(self, method_name, *data):
        # Check data types, parameters, etc.
        return PatchingConditionsChain("sklearnex.algorithm")

    def _onedal_gpu_supported(self, method_name, *data):
        # Check GPU-specific requirements
        return PatchingConditionsChain("sklearnex.algorithm.gpu")
```

**Dispatch Flow:**
1. Check `_onedal_gpu_supported()` → Use GPU oneDAL
2. Check `_onedal_cpu_supported()` → Use CPU oneDAL
3. Fallback → Use original sklearn

## Algorithm Categories

**Supported Algorithms with oneDAL:**
- **Clustering**: DBSCAN, K-Means
- **Linear Models**: LogisticRegression, Ridge, LinearRegression
- **Ensemble**: RandomForestClassifier/Regressor
- **Decomposition**: PCA, IncrementalPCA
- **Neighbors**: KNeighborsClassifier/Regressor
- **SVM**: SVC, SVR, NuSVC, NuSVR

**GPU Support Status:**
- **Full GPU**: DBSCAN, K-Means, PCA, KNeighbors
- **Limited GPU**: LogisticRegression (2024.1+), SVM
- **CPU Only**: RandomForest, Ridge, IncrementalPCA

## Key Implementation Files
- `sklearnex/dispatcher.py:36` - `get_patch_map_core()` function
- `sklearnex/_device_offload.py:72` - `dispatch()` function
- `sklearnex/_config.py` - Configuration API
- `sklearnex/base.py` - oneDALEstimator base class

## Distributed Computing (SPMD)
**Location**: `sklearnex/spmd/`
**Usage**: Same API, distributed across MPI nodes
```python
from sklearnex.spmd.cluster import DBSCAN  # Distributed version
```

## Preview Features
**Activation**: `export SKLEARNEX_PREVIEW=1`
**Location**: `sklearnex/preview/`
**Content**: Experimental algorithms, enhanced covariance, advanced PCA

## Error Handling
**Fallback Chain**: oneDAL GPU → oneDAL CPU → sklearn → Error

**Common Fallback Triggers:**
- Sparse data (most algorithms don't support)
- Unsupported parameters
- GPU memory limits
- Wrong data types
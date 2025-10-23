# AGENTS.md - oneDAL Backend (onedal/)

## Purpose
Low-level Python bindings to Intel oneDAL using pybind11, providing CPU/GPU execution and memory management.

## Key Components
- `__init__.py` - Backend selection (DPC++/Host)
- `_config.py` - Thread-local configuration
- `_device_offload.py` - Device dispatch utilities
- `common/` - Core infrastructure and policies
- `datatypes/` - Data conversion (NumPy, SYCL USM, DLPack)
- Algorithm modules: `cluster/`, `linear_model/`, `decomposition/`, etc.

## Backend System
```python
# Automatic backend selection
try:
    import onedal._onedal_py_dpc  # GPU backend
except ImportError:
    import onedal._onedal_py_host  # CPU backend
```

## Configuration
```python
from onedal import config_context

# GPU acceleration
with config_context(target_offload="gpu:0"):
    model.fit(X, y)

# Auto device selection (default)
with config_context(target_offload="auto"):
    model.fit(X, y)  # Uses data location to choose device
```

## Data Conversion
- **NumPy**: Zero-copy conversion via `to_table()`
- **SYCL USM**: GPU memory sharing (`__sycl_usm_array_interface__`)
- **DLPack**: Cross-framework tensor exchange

## Algorithm Categories
- **Clustering**: DBSCAN, K-Means
- **Linear Models**: Linear/Ridge/Logistic regression
- **Decomposition**: PCA, Incremental PCA
- **SVM**: SVC, SVR with kernel methods
- **Ensemble**: Random Forest
- **Statistics**: Basic statistics, covariance

## For AI Agents
- Use `config_context` for device selection
- Prefer zero-copy operations with `to_table()`
- Handle CPU/GPU fallback gracefully
- Monitor memory usage on GPU
- Test across different device configurations
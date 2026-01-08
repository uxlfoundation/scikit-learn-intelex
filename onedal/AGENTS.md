# AGENTS.md - oneDAL Backend (onedal/)

## Purpose
Low-level Python bindings to Intel oneDAL using pybind11, providing CPU/GPU execution and memory management.

## Key Components
- `__init__.py` - Backend selection (DPC++/Host)
- `_config.py` - Thread-local configuration
- `_device_offload.py` - Device dispatch utilities
- `common/` - Core infrastructure and policies
- `datatypes/` - Data conversion (NumPy, SYCL USM, DLPack)
- Algorithm modules: `cluster/`, `linear_model/`, `decomposition/`, `svm/`, `ensemble/`, `basic_statistics/`, `covariance/`

## Backend System
Automatic backend selection between GPU (DPC++) and CPU (Host) based on available hardware and dependencies.

## Data Conversion
- **NumPy**: Zero-copy conversion for CPU arrays
- **SYCL USM**: GPU memory sharing via `__sycl_usm_array_interface__`
- **DLPack**: Cross-framework tensor exchange

## Algorithm Categories
- **Clustering**: DBSCAN, K-Means
- **Linear Models**: Linear/Ridge/Logistic regression
- **Decomposition**: PCA, Incremental PCA
- **SVM**: SVC, SVR with kernel methods
- **Ensemble**: Random Forest
- **Statistics**: Basic statistics, covariance

## For AI Agents
- Prefer zero-copy operations with `to_table()`
- Handle CPU/GPU backend availability gracefully
- Monitor memory usage on GPU
- Test across different device configurations

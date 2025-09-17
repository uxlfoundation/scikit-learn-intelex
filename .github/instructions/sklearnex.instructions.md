# sklearnex/* - Primary sklearn-compatible Interface

## Purpose
Primary user interface for sklearn acceleration with patching system and device offloading.

## Key Files & Functions
- `dispatcher.py`: Patching system (`get_patch_map_core` line 36)
- `_device_offload.py`: GPU/CPU dispatch (`dispatch` function line 72)
- `_config.py`: Global configuration (target_offload, allow_fallback_to_host)
- `base.py`: oneDALEstimator base class for all accelerated algorithms

## Usage Patterns

**Global Patching (Most Common):**
```python
from sklearnex import patch_sklearn
patch_sklearn()                      # All sklearn imports now accelerated
from sklearn.cluster import DBSCAN   # Uses oneDAL implementation
```

**Selective Patching:**
```python
patch_sklearn(["DBSCAN", "KMeans"])  # Only specific algorithms
```

**Direct Import (No Patching):**
```python
from sklearnex.cluster import DBSCAN  # Always oneDAL implementation
```

**Device Control:**
```python
from sklearnex import config_context

# GPU acceleration (requires Intel GPU + drivers)
with config_context(target_offload="gpu:0"):
    model.fit(X, y)

# Force CPU
with config_context(target_offload="cpu"):
    model.fit(X, y)
```

## Testing
```bash
# sklearnex-specific tests
pytest --verbose --pyargs sklearnex
pytest sklearnex/tests/test_patching.py       # Core patching functionality
pytest sklearnex/tests/test_config.py         # Configuration system
```

## Development Notes
- All sklearn-compatible algorithms inherit from `base.oneDALEstimator`
- Fallback to original sklearn if oneDAL implementation unavailable
- Device offloading requires Intel GPU drivers and SYCL runtime
- See `sklearnex/AGENTS.md` for detailed module information
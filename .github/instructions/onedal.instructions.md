# onedal/* - Low-Level C++ Bindings

## Purpose
Pybind11-based C++ bindings providing the bridge between Python and Intel oneDAL C++ library.

## Key Components
- `datatypes/`: Memory management and array conversions (NumPy, SYCL USM, DLPack)
- `common/`: Policy management, device selection, serialization
- `*/`: Algorithm-specific implementations (cluster/, decomposition/, linear_model/, etc.)
- `spmd/`: Distributed computing interfaces

## Memory Management
```python
# Zero-copy conversions handled automatically
import numpy as np
from onedal.cluster import DBSCAN

# NumPy arrays converted to oneDAL tables without copying
X = np.random.random((1000, 10))
model = DBSCAN().fit(X)  # Automatic NumPy → oneDAL conversion
```

## Device Context

For comprehensive device management, see [onedal/AGENTS.md](../onedal/AGENTS.md).

```python
import dpctl
with dpctl.device_context("gpu:0"):
    model = DBSCAN().fit(X)
```

## Algorithm Structure
- Each algorithm module follows consistent pattern:
  - `fit()` method for training
  - `predict()` method for inference (where applicable)
  - Parameters match oneDAL C++ API
  - Results as Python objects with named attributes

## Testing
```bash
# Low-level onedal tests
pytest onedal/tests/                           # Core functionality
pytest onedal/datatypes/tests/                 # Memory management
pytest onedal/common/tests/                    # Device/policy tests

# Algorithm-specific tests
pytest onedal/cluster/tests/test_dbscan.py     # DBSCAN implementation
pytest onedal/linear_model/tests/              # Linear models
```

## Development Notes
- Direct interface to oneDAL C++ API through pybind11
- Handles memory management between Python/C++ automatically
- Provides foundation for both daal4py and sklearnex layers
- SPMD module enables distributed computing with MPI

## Related Instructions
- `general.instructions.md` - Repository setup and build requirements
- `src.instructions.md` - C++/Cython implementation that uses onedal
- `sklearnex.instructions.md` - High-level layer built on onedal
- `daal4py.instructions.md` - Alternative interface to onedal
- See `onedal/AGENTS.md` for detailed technical implementation
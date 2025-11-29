# AGENTS.md - Core Implementation (src/)

## Purpose
C++/Cython implementation providing direct Python bindings to Intel oneDAL with zero-overhead access, memory management, and distributed computing.

## Key Files
- `daal4py.cpp/.h` - Main C++ interface and NumPy integration
- `npy4daal.h` - NumPy-oneDAL conversion utilities
- `gbt_model_builder.pyx` - Gradient boosting tree builder
- `gettree.pyx` - Tree visitor patterns (sklearn compatibility)
- `transceiver.h` - Communication abstraction for distributed computing
- `dist_*.h` - Distributed algorithm implementations (DBSCAN, K-Means)
- `pickling.h` - Serialization support

## Core Features

### Memory Management
```cpp
// Zero-copy NumPy integration with thread-safe reference counting
class NumpyDeleter : public daal::services::DeleterIface {
    // GIL-protected cleanup of Python objects
};
```

### Distributed Computing
```cpp
// MPI-based communication layer
class transceiver_iface {
    virtual void gather(...) = 0;
    virtual void bcast(...) = 0;
    virtual void reduce_all(...) = 0;
};
```

### Tree Model Building
```cython
# Cython interface for external model conversion
cdef class gbt_classification_model_builder:
    def create_tree(self, n_nodes, class_label)
    def add_split(self, feature_index, threshold)
    def add_leaf(self, response, cover)
```

## For AI Agents
- src/ contains performance-critical C++/Cython code
- Use existing patterns for memory management (zero-copy, GIL protection)
- Distributed algorithms follow map-reduce patterns
- Model builders enable external framework integration (XGBoost→oneDAL)
- Maintain thread safety and cross-platform compatibility
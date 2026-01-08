# AGENTS.md - Core Implementation (src/)

## Purpose
C++/Cython implementation providing direct Python bindings to Intel oneDAL with zero-overhead access, memory management, and distributed computing.

## Key Files by Function

### Core Bindings
- `daal4py.cpp/.h`: Main C++ interface and NumPy integration
- `npy4daal.h`: NumPy-oneDAL conversion utilities

### Model Builders
- `gbt_model_builder.pyx`: Gradient boosting tree builder (XGBoost/LightGBM/CatBoost conversion)
- `log_reg_model_builder.pyx`: Logistic regression model builder
- `gettree.pyx`: Tree visitor patterns for sklearn compatibility

### Distributed Computing
- `transceiver.h/.cpp`: Communication abstraction layer
- `dist_*.h`: Distributed algorithm implementations (DBSCAN, K-Means, Linear Regression, PCA, Covariance)
- `mpi/`: MPI-specific communication primitives

### Utilities
- `pickling.h`: Serialization support for model persistence

## Core Features

### Memory Management
Zero-copy NumPy integration with GIL-protected cleanup and thread-safe reference counting.

### Distributed Computing
MPI-based communication layer with gather, broadcast, and reduce operations. Map-reduce patterns for distributed algorithms.

### Model Building
Cython interfaces for converting external ML framework models to oneDAL format. Supports tree construction, node splits, and leaf assignments.

## For AI Agents
- Performance-critical C++/Cython code requiring compilation
- Zero-copy memory patterns essential for performance
- GIL protection required for Python object manipulation
- Distributed algorithms use MPI communication layer
- Model builders enable XGBoost/LightGBM/CatBoost integration
- Maintain thread safety and cross-platform compatibility

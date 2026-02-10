# AGENTS.md - daal4py Layer

## Purpose
Native Python interface to oneDAL with three APIs: native oneDAL algorithms, sklearn-compatible wrappers, and model builders for external ML frameworks.

## Three APIs

### 1. Native oneDAL API
Direct access to oneDAL algorithms with explicit setup/compute/finalize phases. Provides maximum control and performance. Import via `import daal4py as d4p`.

### 2. sklearn-Compatible API
Drop-in replacements for sklearn estimators. Located in `daal4py/sklearn/`. Monkeypatch system provides transparent acceleration via `patch_sklearn()`.

### 3. Model Builders API
Convert trained models from XGBoost, LightGBM, CatBoost to oneDAL format for accelerated inference. Located in `daal4py/mb/`. Provides 10-100X speedup (varies by model and data).

## Key Components

### Core Files
- `__init__.py`: Native API entry, algorithm loading
- `sklearn/`: sklearn-compatible implementations
- `sklearn/monkeypatch/dispatcher.py`: Patching system
- `mb/tree_based_builders.py`: Tree model conversions
- `mb/logistic_regression_builders.py`: LogReg conversions

### Algorithm Categories
- **Clustering**: dbscan, kmeans
- **Classification**: adaboost, brownboost, decision_forest, decision_tree, gbt_classification, logitboost, naive_bayes, stump, svm
- **Regression**: decision_forest, decision_tree, elastic_net, gbt_regression, lasso, linear, ridge, stump
- **Decomposition**: pca, svd
- **Statistics**: covariance, low_order_moments, correlation_distance, cosine_distance
- **Other**: association_rules, bacon_outlier, cholesky, em_gmm, implicit_als, knn, normalization, qr, pivoted_qr, quantiles, sorting, outlier_detection

## Monkeypatch System

### Implementation
Located in `daal4py/sklearn/monkeypatch/dispatcher.py`. Replaces sklearn estimators with daal4py implementations when conditions met.

### Core Functions
- `patch_sklearn()`: Replace sklearn algorithms globally
- `unpatch_sklearn()`: Restore original sklearn
- `get_patch_map()`: Retrieve algorithm mappings
- Condition checking via `_daal*_check_supported()` functions

### Patching Logic
Checks data characteristics (density, dtype, shape), algorithm parameters, and oneDAL compatibility before applying acceleration.

## Model Builders

Convert externally trained models to oneDAL format for **10-100X faster inference** (speedup varies by model complexity and data size).

### Supported Frameworks

**Tree-Based Models:**
- **XGBoost**: `XGBClassifier`, `XGBRegressor`, `Booster` objects
- **LightGBM**: `LGBMClassifier`, `LGBMRegressor`, `Booster` objects
- **CatBoost**: `CatBoostClassifier`, `CatBoostRegressor` objects
- **TreeLite**: For sklearn `HistGradientBoostingClassifier/Regressor` conversion

**Linear Models:**
- **sklearn LogisticRegression**: Binary and multinomial classification
- **sklearn SGDClassifier**: Linear classification models

### Usage Pattern

```python
from daal4py.mb import convert_model
import xgboost as xgb

# 1. Train with external framework
xgb_model = xgb.XGBClassifier().fit(X_train, y_train)

# 2. Convert to oneDAL
daal_model = convert_model(xgb_model)

# 3. Accelerated inference (10-100X faster)
predictions = daal_model.predict(X_test)
```

### Features
- **No accuracy loss**: Exact prediction equivalence with original models
- **SHAP support**: Accelerated SHAP value computation for tree models
- **Batch prediction**: Optimized for large-scale inference workloads
- **Memory efficient**: Optimized internal representation

### Implementation
- Tree builders: `mb/gbt_convertors.py`, `mb/tree_based_builders.py`
- LogReg builders: `mb/logistic_regression_builders.py`
- Examples: `examples/mb/model_builders_*.py`

## Distributed Computing (SPMD)

### Architecture
MPI-based Single Program Multiple Data execution. Each rank processes data partition, results aggregated at master.

### Implementation
Core logic in `src/dist_*.h` files. Python entry via mpi4py with `daalinit()` and `daalfini()`.

### Supported Algorithms
DBSCAN, K-Means, Linear Regression, Ridge Regression, PCA, Covariance, Low Order Moments, Naive Bayes, QR Decomposition, SVD

### Requirements
MPI installation (Intel MPI or MPICH) and mpi4py package. Launch via `mpirun -n <ranks>`.

## Performance Optimization

### Memory Patterns
- Zero-copy for NumPy contiguous arrays
- Automatic data type conversion when needed
- C-contiguous preferred over Fortran-contiguous
- Direct array access via `make2d()` utility

### Threading
Intel TBB parallelism across cores. Control via daal4py threading functions.

### GPU Acceleration
Limited support via oneDAL backend. Most algorithms CPU-only; GPU support primarily in sklearnex layer.

### Data Requirements
Dense data required for most algorithms. Limited sparse support (CSR format) for some.

## Integration Architecture

### daal4py and onedal - Parallel Bindings
daal4py and onedal are **separate** Python binding implementations to oneDAL C++:
- **daal4py**: Cython-based bindings (legacy, src/_daal4py C extension)
- **onedal**: pybind11-based bindings (modern, GPU-optimized)
- **sklearnex**: Uses both - onedal for GPU/modern algorithms, daal4py for legacy compatibility

### Data Flow
**daal4py path**: NumPy → Cython bindings → oneDAL C++ → Results
**onedal path**: NumPy → pybind11 bindings → oneDAL C++ → Results
**sklearnex**: Chooses appropriate path based on algorithm and device

### Error Handling
- Input validation in Python layer
- C++ exceptions converted to Python exceptions
- Automatic fallback to sklearn in monkeypatch system when conditions not met

## Development Guidelines

### Adding New Algorithms
1. Check if oneDAL C++ algorithm exists
2. Add to generator/wrappers.py if available
3. Rebuild to generate bindings
4. Add sklearn wrapper in `daal4py/sklearn/` if needed
5. Update monkeypatch dispatcher for sklearn compatibility

### Modifying Existing Algorithms
- Native API: Modify generated sources or generator templates
- sklearn API: Direct edits in `daal4py/sklearn/`
- Model builders: Edit `daal4py/mb/`

### Testing
Run pytest on `daal4py/sklearn/` and `tests/` for validation. MPI tests require `mpirun -n 4`.

## Algorithm Decision Matrix

**Use Native daal4py when:**
- Maximum performance needed
- Advanced algorithm control required
- Using distributed (SPMD) mode

**Use sklearn-compatible API when:**
- Drop-in sklearn replacement desired
- Gradual migration from sklearn
- Compatibility with sklearn ecosystem required

**Use Model Builders when:**
- Already trained XGBoost/LightGBM/CatBoost models
- Inference performance critical
- Deployment optimization needed

## Key Implementation Patterns
- Native API provides explicit control, sklearn API provides compatibility
- Model builders accelerate inference of externally trained models
- Monkeypatch system enables transparent acceleration via condition checking
- SPMD mode requires MPI for distributed execution across multiple nodes
- Check algorithm availability in oneDAL C++ before attempting wrapper
- Generated code in build directories, templates in generator/
- Zero-copy operations critical for performance
- Dense data and contiguous arrays required for most algorithms

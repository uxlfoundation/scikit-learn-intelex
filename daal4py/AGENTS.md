# AGENTS.md - daal4py Package

## Purpose
**Direct Python bindings to Intel oneDAL** for maximum performance

## Two APIs
1. **Native oneDAL**: `import daal4py as d4p`
2. **sklearn-compatible**: `from daal4py.sklearn import ...`
3. **Model Builders**: `from daal4py.mb import convert_model`

## Native oneDAL API Usage

**Basic Pattern:**
```python
import daal4py as d4p
import numpy as np

# Create algorithm
algorithm = d4p.dbscan(epsilon=0.5, minObservations=5)

# Run computation
result = algorithm.compute(data)

# Access results
cluster_labels = result.assignments
core_indices = result.coreIndices
```

**Common Algorithms:**
```python
# Clustering
d4p.dbscan(epsilon=0.5, minObservations=5)
d4p.kmeans(nClusters=3, maxIterations=300)

# Decomposition
d4p.pca(method="defaultDense")
d4p.svd(method="defaultDense")

# Linear Models
d4p.linear_regression_training()
d4p.ridge_regression_training(ridgeParameters=1.0)
```

## sklearn-Compatible API

**Usage:**
```python
from daal4py.sklearn.cluster import DBSCAN
from daal4py.sklearn.linear_model import Ridge

# Use like normal sklearn
clusterer = DBSCAN(eps=0.5, min_samples=5)
labels = clusterer.fit_predict(X)
```

**Patching System:**
```python
from daal4py.sklearn.monkeypatch import patch_sklearn
patch_sklearn()  # Replace sklearn algorithms with daal4py versions
```

## Model Builders (`mb/`)

**Purpose**: Convert external ML models to oneDAL for faster inference

**Supported Frameworks:**
```python
from daal4py.mb import convert_model

# XGBoost/LightGBM/CatBoost → oneDAL
externalModel = xgb.XGBClassifier().fit(X, y)
d4p_model = convert_model(externalModel)

# Use oneDAL for fast prediction
predictions = d4p_model.predict(X_test)
prob = d4p_model.predict_proba(X_test)
```

**Benefits**: 10-100x faster inference than original models

### 3. Monkeypatch System (`sklearn/monkeypatch/`)

**Purpose**: Original patching mechanism for scikit-learn replacement

**Core Implementation** (`dispatcher.py:57-200`):
```python
@lru_cache(maxsize=None)
def _get_map_of_algorithms():
    mapping = {
        "pca": [[(decomposition_module, "PCA", PCA_daal4py), None]],
        "kmeans": [[(cluster_module, "KMeans", KMeans_daal4py), None]],
        "dbscan": [[(cluster_module, "DBSCAN", DBSCAN_daal4py), None]],
        # ... complete algorithm mapping
    }
    return mapping
```

**Patching Functions**:
- `patch_sklearn()`: Replace sklearn algorithms with daal4py versions
- `unpatch_sklearn()`: Restore original sklearn implementations
- `get_patch_map()`: Retrieve current algorithm mappings
- `enable_patching()`: Context-based patching control

**Condition Checking**:
```python
def _daal4py_check_supported(estimator, method_name, *data):
    # Check data characteristics (density, dtypes, shape)
    # Check algorithm parameters
    # Check oneDAL version compatibility
    # Return boolean + condition chain
```

### 4. Model Builders (`mb/`)

**Purpose**: Convert external ML library models to oneDAL for accelerated inference

#### Tree-Based Models (`tree_based_builders.py`)

**Supported Libraries**:
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Microsoft gradient boosting
- **CatBoost**: Yandex gradient boosting
- **Treelite**: Universal tree model format

**Implementation Pattern**:
```python
class GBTDAALModel(GBTDAALBaseModel):
    def __init__(self, model):
        # 1. Extract model parameters and structure
        # 2. Convert to oneDAL tree format
        # 3. Create oneDAL inference model

    def predict(self, X):
        # Use oneDAL optimized prediction

    def predict_proba(self, X):
        # Probabilistic predictions for classification
```

**Conversion Process**:
1. **Tree Extraction**: Parse external model tree structures
2. **Parameter Mapping**: Convert hyperparameters to oneDAL format
3. **Model Creation**: Build oneDAL gradient boosting model
4. **Validation**: Verify numerical equivalence with original model

#### Logistic Regression Models (`logistic_regression_builders.py`)

**Supported Sources**:
- sklearn LogisticRegression (binary/multinomial)
- sklearn SGDClassifier (with log loss)
- Direct coefficient specification

**Features**:
- Binary and multinomial classification
- Coefficient and intercept preservation
- oneDAL optimized prediction pipeline

### 5. Distributed Computing (SPMD)

**Purpose**: Single Program Multiple Data parallel processing across multiple nodes

**Implementation Location**:
- C++ Headers: `src/dist_*.h` files
- Examples: `examples/daal4py/*_spmd.py`

**Architecture**:
```cpp
// C++ distributed computing framework (src/dist_custom.h)
template <typename T1, typename T2>
class dist {
    // MPI communication primitives
    // Data serialization/deserialization
    // Distributed algorithm coordination
};
```

**Supported Algorithms**:
- **DBSCAN**: `dist_dbscan.h` - Distributed density clustering
- **K-Means**: `dist_kmeans.h` - Distributed centroid-based clustering
- **Linear Regression**: Distributed least squares
- **PCA**: Distributed principal component analysis
- **Covariance**: Distributed covariance matrix computation

**SPMD Usage Pattern**:
```python
import daal4py as d4p

# Initialize distributed backend
d4p.daalinit()

# Distributed algorithm execution
result = algorithm.compute(local_data_chunk)

# Finalize and collect results
d4p.daalfini()
```

**MPI Integration**:
- Automatic rank and size detection
- Efficient data distribution strategies
- Collective communication operations
- Fault tolerance and load balancing

## Performance Optimization Strategies

### 1. Memory Management

**Zero-Copy Operations**:
- Direct NumPy array access via `make2d()` utility
- In-place data transformations where possible
- Efficient C++ ↔ Python data exchange

**Memory Layout Optimization**:
```python
# Efficient data preparation (daal4py/sklearn/_utils.py)
def make2d(X):
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return np.ascontiguousarray(X, dtype=np.float64)
```

### 2. Algorithmic Optimizations

**Solver Selection**:
- Analytical solutions for overdetermined systems
- Iterative methods for large-scale problems
- Specialized algorithms for sparse data

**Parallel Execution**:
- Intel TBB threading for shared-memory parallelism
- MPI for distributed-memory parallelism
- Vectorization via Intel SIMD instructions

### 3. Data Type Optimization

**Precision Selection**:
```python
def getFPType(X):
    """Determine optimal floating-point precision"""
    if hasattr(X, 'dtype'):
        if X.dtype == np.float32:
            return "float"
        else:
            return "double"
    return "double"  # Default to double precision
```

### 4. Condition-Based Optimization

**Patching Conditions** (Pattern across all algorithms):
```python
def _daal4py_supported(self, method_name, *data):
    conditions = PatchingConditionsChain("daal4py.algorithm.method")

    # Data characteristics
    conditions.and_condition(not sp.issparse(data[0]), "Sparse not supported")
    conditions.and_condition(data[0].dtype in [np.float32, np.float64], "Invalid dtype")

    # Algorithm parameters
    conditions.and_condition(self.metric == "euclidean", "Only euclidean metric")
    conditions.and_condition(self.algorithm == "auto", "Algorithm must be auto")

    return conditions
```

## Integration Architecture

### With oneDAL C++ Library

**Direct Binding Layer**:
- Cython-based C++ wrapper generation
- Template instantiation for algorithm variants
- Exception handling and error propagation
- Memory management coordination

**Algorithm Instantiation Pattern**:
```cpp
// C++ algorithm instantiation (generated via Cython)
daal::algorithms::dbscan::Batch<float, daal::algorithms::dbscan::defaultDense> algorithm;
algorithm.parameter.epsilon = eps;
algorithm.parameter.minObservations = min_samples;
algorithm.input.set(daal::algorithms::dbscan::data, numericTable);
daal::algorithms::dbscan::ResultPtr result = algorithm.compute();
```

### With sklearnex Package

**Layered Architecture**:
1. **sklearnex**: High-level API with device offloading
2. **daal4py**: Core algorithms and patching
3. **oneDAL**: Low-level optimized implementations

**API Delegation**:
```python
# sklearnex delegates to daal4py for compatible cases
if _is_daal4py_supported():
    return daal4py_algorithm.fit(X, y)
else:
    return sklearn_algorithm.fit(X, y)
```

### With External Libraries

**Model Conversion Pipeline**:
```python
# XGBoost → oneDAL conversion example
def get_gbt_model_from_xgboost(xgb_model):
    # 1. Extract XGBoost JSON representation
    # 2. Parse tree structures and parameters
    # 3. Convert to oneDAL tree format
    # 4. Create oneDAL gradient boosting model
    # 5. Return optimized prediction interface
```

## Error Handling and Fallbacks

### Exception Management

**oneDAL Error Handling**:
- C++ exception translation to Python
- Detailed error messages with context
- Graceful degradation to sklearn when possible

**Common Error Patterns**:
```python
try:
    result = daal4py_algorithm.compute(data)
except RuntimeError as e:
    if "not supported" in str(e):
        # Fallback to sklearn
        return sklearn_algorithm.fit(X, y)
    else:
        raise
```

### Validation and Checks

**Input Validation**:
- Data type and shape verification
- Parameter range checking
- Memory layout validation
- Feature name consistency

**Compatibility Checking**:
- oneDAL version requirements
- Algorithm parameter support
- Hardware capability detection

## Development Guidelines

### Adding New Algorithms

1. **Create Native Wrapper**:
   ```python
   def _daal_algorithm(X, y=None, **params):
       # Convert inputs to oneDAL format
       # Configure oneDAL algorithm
       # Execute computation
       # Convert results to expected format
   ```

2. **Implement sklearn Interface**:
   ```python
   class Algorithm(sklearn_Algorithm):
       def fit(self, X, y=None):
           return self._daal_fit(X, y)
   ```

3. **Add to Dispatcher**:
   ```python
   # Update monkeypatch/dispatcher.py
   mapping["algorithm"] = [[(module, "Algorithm", Algorithm_daal4py), None]]
   ```

4. **Create Tests**:
   ```python
   # Numerical accuracy tests
   # Performance benchmarks
   # Edge case validation
   ```

### Performance Optimization Guidelines

- **Minimize Data Copies**: Use views and in-place operations
- **Leverage oneDAL Optimizations**: Choose appropriate algorithms and parameters
- **Profile Memory Usage**: Monitor peak memory consumption
- **Validate Numerically**: Ensure mathematical correctness
- **Benchmark Performance**: Measure against sklearn baselines

### Distributed Computing Guidelines

- **Design for Scalability**: Consider communication overhead
- **Handle Data Distribution**: Implement efficient partitioning
- **Manage Dependencies**: Coordinate between nodes
- **Test at Scale**: Validate with realistic data sizes

## File Location Reference

### Core Implementation
- `daal4py/__init__.py:53-73` - Core binding imports and initialization
- `daal4py/sklearn/monkeypatch/dispatcher.py:57-200` - Algorithm mapping system
- `src/daal4py.cpp` - Main C++/Cython implementation
- `src/dist_*.h` - Distributed computing headers

### Algorithm Examples
- `daal4py/sklearn/cluster/dbscan.py:35-56` - DBSCAN oneDAL integration
- `daal4py/sklearn/linear_model/_linear.py` - Linear regression implementation
- `daal4py/sklearn/decomposition/_pca.py` - PCA with oneDAL optimization

### Model Builders
- `daal4py/mb/tree_based_builders.py:65-200` - GBT model conversion
- `daal4py/mb/logistic_regression_builders.py` - LogReg model conversion
- `daal4py/mb/gbt_convertors.py` - External library integration

### Distributed Computing
- `examples/daal4py/*_spmd.py` - SPMD usage examples
- `src/dist_dbscan.h:28-100` - Distributed DBSCAN implementation
- `src/mpi/` - MPI communication layer

## AI Agent Development Guidelines

When working with daal4py, AI agents should:

1. **Understand the Native API**: Recognize direct oneDAL algorithm access patterns
2. **Respect Performance Requirements**: Maintain zero-copy operations where possible
3. **Handle Distributed Computing**: Account for MPI coordination and data distribution
4. **Validate Numerically**: Ensure algorithmic correctness against sklearn
5. **Consider Memory Constraints**: Monitor memory usage in large-scale scenarios
6. **Test Across Platforms**: Validate on different hardware configurations
7. **Document Performance**: Clearly specify optimization benefits and limitations
8. **Maintain Compatibility**: Preserve sklearn API contracts and behavior

The daal4py package represents the performance-critical foundation of the Intel Extension for Scikit-learn, providing both the algorithmic engine and the compatibility layer that enables seamless acceleration of existing scikit-learn workflows.
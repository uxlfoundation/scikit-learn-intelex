# daal4py/* - Direct oneDAL Python Bindings

## Purpose
Direct Python bindings to Intel oneDAL for maximum performance and model builders for XGBoost/LightGBM conversion.

## Three Sub-APIs
1. **Native oneDAL**: `import daal4py as d4p` - Direct algorithm access
2. **sklearn-compatible**: `from daal4py.sklearn import ...` - sklearn API with oneDAL backend
3. **Model Builders**: `from daal4py.mb import convert_model` - External model conversion

## Native oneDAL Pattern
```python
import daal4py as d4p
import numpy as np

# Create algorithm with parameters
algorithm = d4p.dbscan(epsilon=0.5, minObservations=5)

# Run computation
result = algorithm.compute(data)

# Access results (algorithm-specific attributes)
cluster_labels = result.assignments
core_indices = result.coreIndices
```

## Common Native Algorithms
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

## Model Builders (mb/)
```python
from daal4py.mb import convert_model

# Convert external models to oneDAL format
d4p_model = convert_model(xgb_model)      # XGBoost → oneDAL
d4p_model = convert_model(lgb_model)      # LightGBM → oneDAL
d4p_model = convert_model(catboost_model) # CatBoost → oneDAL

# Use converted model for fast inference
predictions = d4p_model.compute(test_data)
```

## Testing
```bash
# Native daal4py tests
pytest --verbose --pyargs daal4py
pytest tests/test_daal4py_examples.py         # Native API examples
pytest tests/test_model_builders.py           # Model conversion tests

# sklearn compatibility in daal4py
pytest daal4py/sklearn/tests/                 # sklearn-compatible API
```

## Development Notes
- Native API provides direct oneDAL algorithm access (fastest performance)
- sklearn-compatible API in `daal4py/sklearn/` maintains full sklearn compatibility
- Model builders enable oneDAL inference for models trained with other frameworks
- See `daal4py/AGENTS.md` for detailed algorithm usage patterns
# AGENTS.md - Examples (examples/)

## Purpose
113 Python scripts and 19 Jupyter notebooks demonstrating Intel Extension for Scikit-learn usage patterns.

## Directory Structure
- `daal4py/` - Native oneDAL API examples (80+ scripts)
- `sklearnex/` - Accelerated sklearn examples (25+ scripts)
- `mb/` - Model builder examples (XGBoost/LightGBM/CatBoost conversion)
- `notebooks/` - Jupyter tutorials with real datasets
- `utils/` - Utility functions

## Key Usage Patterns

### Native oneDAL API
```python
import daal4py as d4p
algorithm = d4p.dbscan(epsilon=0.5, minObservations=5)
result = algorithm.compute(data)
```

### Accelerated sklearn
```python
from sklearnex import patch_sklearn
patch_sklearn()  # All sklearn imports now accelerated
from sklearn.cluster import DBSCAN
```

### GPU Acceleration
```python
from sklearnex import config_context
with config_context(target_offload="gpu:0"):
    model.fit(X, y)
```

### Distributed Computing
```python
import daal4py as d4p
d4p.daalinit()  # Initialize MPI
# ... distributed computation
d4p.daalfini()  # Cleanup
```

### Model Conversion
```python
from daal4py.mb import convert_model
d4p_model = convert_model(xgb_model)  # 10-100x faster inference
```

## Algorithm Categories
- **Clustering**: DBSCAN, K-Means
- **Linear Models**: Linear/Ridge/Logistic regression
- **Ensemble**: Random Forest, Gradient boosting
- **Decomposition**: PCA, SVD
- **Statistics**: Moments, covariance
- **SVM**: Classification and regression

## For AI Agents
- Use examples as templates for new implementations
- Follow established patterns for performance optimization
- Include both sklearn and oneDAL performance comparisons
- Test examples across CPU/GPU configurations
# daal4py/* - Direct oneDAL Python Bindings

## Purpose
Direct Python bindings to Intel oneDAL for maximum performance and model builders for XGBoost/LightGBM conversion.

## Three Sub-APIs
1. **Native oneDAL**: `import daal4py as d4p` - Direct algorithm access
2. **sklearn-compatible**: `from daal4py.sklearn import ...` - sklearn API with oneDAL backend
3. **Model Builders**: `from daal4py.mb import convert_model` - External model conversion

## API Overview

For detailed native oneDAL patterns and model builders, see [daal4py/AGENTS.md](../daal4py/AGENTS.md).

**Basic Pattern**:
```python
import daal4py as d4p
algorithm = d4p.dbscan(epsilon=0.5, minObservations=5)
result = algorithm.compute(data)
```

**Model Conversion**:
```python
from daal4py.mb import convert_model
d4p_model = convert_model(xgb_model)  # 10-100x faster inference
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

## Related Instructions
- `general.instructions.md` - Repository setup and build requirements
- `onedal.instructions.md` - Low-level backend that daal4py wraps
- `src.instructions.md` - Core C++/Cython implementation details
- `tests.instructions.md` - Testing native oneDAL algorithms
- See `daal4py/AGENTS.md` for detailed algorithm usage patterns
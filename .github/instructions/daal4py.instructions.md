# daal4py/* - Direct oneDAL Python Bindings

## Purpose
Direct Python bindings to oneDAL for maximum performance and model builders for XGBoost/LightGBM/CatBoost conversion.

## Three Sub-APIs
1. **Native oneDAL**: `import daal4py as d4p` - Direct algorithm access with explicit compute phases
2. **sklearn-compatible**: `from daal4py.sklearn import ...` - sklearn API with oneDAL backend
3. **Model Builders**: `from daal4py.mb import convert_model` - External model conversion for accelerated inference

## Testing
```bash
pytest --verbose daal4py
pytest tests/test_daal4py_examples.py
pytest tests/test_model_builders.py
pytest daal4py/sklearn/tests/
```

## For GitHub Copilot

See [daal4py/AGENTS.md](../daal4py/AGENTS.md) for comprehensive information including:
- Detailed native oneDAL API patterns
- Model builder conversion process
- Monkeypatch system architecture
- Distributed computing (SPMD) implementation
- Performance optimization strategies

## Related Instructions
- `general.instructions.md` - Repository setup
- `onedal.instructions.md` - Low-level backend
- `src.instructions.md` - C++/Cython implementation
- `tests.instructions.md` - Testing native algorithms

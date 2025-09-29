# AGENTS.md - Testing Infrastructure (tests/)

## Purpose
Comprehensive validation infrastructure ensuring numerical accuracy, performance compliance, and cross-platform reliability.

## Key Test Modules
- `test_daal4py_examples.py` - Native daal4py algorithm validation
- `test_model_builders.py` - External framework integration (XGBoost/LightGBM)
- `test_daal4py_spmd_examples.py` - Distributed computing validation
- `test_estimators.py` - sklearn compatibility validation
- `test_npy.py` - NumPy data type validation
- `run_examples.py` - Cross-platform example execution
- `unittest_data/` - Reference datasets for validation

## Validation Patterns

### Numerical Accuracy
```python
# Standard tolerance for floating-point comparisons
np.testing.assert_allclose(actual, expected, atol=1e-05)

# Matrix reconstruction validation (SVD/QR)
np.testing.assert_allclose(original, reconstructed)
```

### Model Builder Testing
```python
# XGBoost conversion accuracy
xgb_predictions = xgb_model.predict(X)
d4p_predictions = convert_model(xgb_model).predict(X)
np.testing.assert_allclose(xgb_predictions, d4p_predictions)
```

### Performance Validation
```python
# Execution time limits
@dataclass
class Config:
    timeout_cpu_seconds: int = 170  # Default (verified in tests/test_daal4py_examples.py)
    # Extended timeouts for complex algorithms
```

### Distributed Testing
```python
# MPI-aware testing with proper rank coordination
@unittest.skipUnless(MPI.COMM_WORLD.size > 1, "Not running in distributed mode")
def test_spmd_algorithm(self):
    # Distributed algorithm validation
```

## Cross-Platform Support
- **OS Detection**: Windows, Linux, macOS compatibility
- **Device Requirements**: CPU/GPU availability checking
- **Dependency Management**: Graceful skipping for missing libraries

## Test Execution Commands

### Local Development Testing
```bash
# Complete test suite (verified in conda-recipe/run_test.sh)
pytest --verbose -s tests/                    # Legacy/integration tests
pytest --verbose --pyargs daal4py            # Native oneDAL API tests
pytest --verbose --pyargs sklearnex          # sklearn compatibility tests
pytest --verbose --pyargs onedal             # Low-level backend tests
pytest --verbose .ci/scripts/test_global_patch.py  # Global patching validation

# With coverage reporting
pytest --cov=onedal --cov=sklearnex --cov-config=.coveragerc --cov-branch
```

### Distributed (SPMD) Testing
```bash
# Requires MPI setup and NO_DIST!=1
mpirun -n 4 python tests/helper_mpi_tests.py \
    pytest -k spmd --with-mpi --verbose --pyargs sklearnex

mpirun -n 4 python tests/helper_mpi_tests.py \
    pytest --verbose -s tests/test_daal4py_spmd_examples.py
```

### Performance Validation
```python
# Timeout configuration patterns (from test_daal4py_examples.py)
@dataclass
class Config:
    timeout_cpu_seconds: int = 170  # Default timeout
    # Algorithm-specific overrides:
    # - gradient_boosted_classification: 480s
    # - complex algorithms: extended timeouts
```

### Dependencies and Platform Testing
```python
# Graceful dependency handling (from run_examples.py)
def has_deps(rule):
    for rule_item in rule:
        try:
            importlib.import_module(rule_item)
        except ImportError:
            return False
    return True

# Platform detection
IS_WIN = plt.system() == "Windows"
IS_MAC = plt.system() == "Darwin"
IS_LIN = plt.system() == "Linux"
```

## For AI Agents
- Use `np.testing.assert_allclose(atol=1e-05)` for numerical validation
- Configure appropriate timeouts based on algorithm complexity
- Handle missing dependencies gracefully with `skipTest()`
- Test both sklearn compatibility and numerical accuracy
- Validate model conversion maintains prediction accuracy
- Run distributed tests with `mpirun -n 4` for SPMD algorithms
- Check hardware requirements before GPU tests
- Use coverage reporting for development validation
# AGENTS.md - Testing Infrastructure (tests/)

## Purpose
Comprehensive validation infrastructure ensuring numerical accuracy, performance compliance, and cross-platform reliability.

## Key Test Modules
- `test_daal4py_examples.py`: Native daal4py algorithm validation
- `test_model_builders.py`: External framework integration (XGBoost/LightGBM/CatBoost)
- `test_daal4py_spmd_examples.py`: Distributed computing validation
- `test_estimators.py`: sklearn compatibility validation
- `test_npy.py`: NumPy data type validation
- `run_examples.py`: Cross-platform example execution
- `helper_mpi_tests.py`: MPI test coordination
- `unittest_data/`: Reference datasets for validation

## Validation Patterns

### Numerical Accuracy
Standard tolerance: `np.testing.assert_allclose(actual, expected, atol=1e-05)`

### Model Builder Testing
Validate prediction equivalence between original models and converted oneDAL models.

### Performance Validation
Default timeout: 170 seconds per test. Algorithm-specific overrides for complex operations (e.g., gradient boosting: 480s).

### Distributed Testing
MPI-aware testing with rank coordination. Skip tests if not running in distributed mode.

## Cross-Platform Support
- OS detection: Windows, Linux
- Device checking: CPU/GPU availability validation
- Dependency management: Graceful skipping for missing libraries

## Test Execution

### Local Development
Core test suites:
- `pytest --verbose -s tests/` - Legacy/integration tests
- `pytest --verbose --pyargs daal4py` - Native oneDAL API tests
- `pytest --verbose --pyargs sklearnex` - sklearn compatibility tests
- `pytest --verbose --pyargs onedal` - Low-level backend tests
- `pytest --verbose .ci/scripts/test_global_patch.py` - Global patching validation

Coverage reporting:
- `pytest --cov=onedal --cov=sklearnex --cov-config=.coveragerc --cov-branch`

### MPI/Distributed (SPMD) Testing
Requirements: MPI installation (Intel MPI or OpenMPI), mpi4py, NO_DIST!=1

Commands:
- `mpirun -n 4 python tests/helper_mpi_tests.py pytest -k spmd --with-mpi --verbose --pyargs sklearnex`
- `mpirun -n 4 python tests/helper_mpi_tests.py pytest --verbose -s tests/test_daal4py_spmd_examples.py`

Validates distributed algorithms: DBSCAN, K-Means, PCA, Linear Regression, Covariance

## For AI Agents
- Use `np.testing.assert_allclose(atol=1e-05)` for numerical validation
- Configure timeouts based on algorithm complexity (default 170s, complex up to 480s)
- Handle missing dependencies with `skipTest()`
- Test both sklearn compatibility and numerical accuracy
- Validate model conversion maintains prediction accuracy
- MPI tests require `mpirun -n 4` and proper MPI setup
- Check hardware availability before GPU tests
- Coverage reporting recommended for development validation

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

**Requirements:** MPI installation (Intel MPI or OpenMPI), mpi4py, NO_DIST!=1

**Setup MPI:**
```bash
# Option 1: Via conda (recommended)
conda install -c conda-forge impi-devel impi_rt mpi4py

# Option 2: System package manager
# Debian/Ubuntu:
sudo apt-get install libopenmpi-dev openmpi-bin
pip install mpi4py

# RedHat/CentOS:
sudo yum install openmpi-devel
pip install mpi4py

# Set MPIROOT if not using conda
export MPIROOT=/path/to/mpi
```

**Rebuild with MPI support:**
```bash
# Ensure NO_DIST is not set
unset NO_DIST
python setup.py build_ext --inplace --force
```

**Run distributed tests:**
```bash
# sklearnex SPMD tests
mpirun -n 4 python tests/helper_mpi_tests.py pytest -k spmd --with-mpi --verbose --pyargs sklearnex

# daal4py SPMD examples
mpirun -n 4 python tests/helper_mpi_tests.py pytest --verbose -s tests/test_daal4py_spmd_examples.py
```

**Validates:** DBSCAN, K-Means, PCA, Linear Regression, Covariance (distributed algorithms)

## scikit-learn Compatibility Testing

### Testing Approach
sklearnex aims for high compatibility with scikit-learn APIs. Tests are run against sklearn's own test suite to validate behavior matches sklearn's implementation.

### Test Coverage
Most sklearn tests pass with sklearnex acceleration. A subset is deselected due to:
1. **Intentional differences**: Performance optimizations that produce mathematically equivalent but not identical results
2. **Unsupported features**: Features not yet implemented in oneDAL backend
3. **Implementation constraints**: Limitations in oneDAL library or SYCL/GPU execution
4. **Platform-specific issues**: Environment-dependent test failures

### Deselected Tests
Configuration: `deselected_tests.yaml` (see file for current count and details)

**Categories** (see `deselected_tests.yaml` for specifics):

**sklearn Version-Specific**:
- sklearn 1.6/1.7 features not yet supported
- Version-specific API changes

**Array API Support**:
- Array API standard compliance differences
- numpy.array_api experimental features
- torch backend incompatibilities

**Algorithm Implementation Differences**:
- PCA: Auto solver selection differs (uses covariance_eigh instead of full)
- RandomForest: Different RNG leading to different feature importances for small tree counts
- SVR: Edge case handling differences (two-sample input)
- KNN: KDTree rare 0-distance point misses

**Unsupported Features**:
- SVM: Subset invariance not yet implemented
- Ridge: Some solver-specific behaviors
- Parameter validation differences

**Platform-Specific**:
- Cache directory access issues on some systems
- Visual Studio build-specific test failures
- Numerical precision differences across platforms

**Exception Handling**:
- Different exception types (but same error conditions)
- Different validation error messages
- oneDAL doesn't throw for non-finite coefficients in some cases

### Version-Specific Deselection
Tests can be deselected conditionally using version specifiers in `deselected_tests.yaml`:
```yaml
- test_name.py::test_function >1.5,<=1.7
```
This deselects only for sklearn versions 1.5.1 through 1.7.x.

### Impact on Users
Deselected tests represent <5% of sklearn's test suite. Most algorithms work identically to sklearn. Differences are:
- Usually in edge cases or rarely-used features
- Documented in test deselection comments
- Tracked for future oneDAL backend improvements

### Running Compatibility Tests
```bash
# Full sklearn compatibility test
pytest --verbose --pyargs sklearnex

# Tests respect deselected_tests.yaml automatically
# To see what's deselected: cat deselected_tests.yaml
```

## For AI Agents
- Use `np.testing.assert_allclose(atol=1e-05)` for numerical validation
- Configure timeouts based on algorithm complexity (default 170s, complex up to 480s)
- Handle missing dependencies with `skipTest()`
- Test both sklearn compatibility and numerical accuracy
- Validate model conversion maintains prediction accuracy
- MPI tests require `mpirun -n 4` and proper MPI setup
- Check hardware availability before GPU tests
- Coverage reporting recommended for development validation

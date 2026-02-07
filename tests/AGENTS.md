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

### Per-Algorithm Tests
In addition to the integration tests above, each algorithm module in `onedal/` and `sklearnex/` has its own `tests/` subdirectory with algorithm-specific tests. For example:
- `sklearnex/basic_statistics/tests/`, `sklearnex/cluster/tests/`, `sklearnex/linear_model/tests/`, etc.
- `onedal/basic_statistics/tests/`, `onedal/cluster/tests/`, `onedal/linear_model/tests/`, etc.

These per-module tests are run via `pytest --verbose sklearnex` and `pytest --verbose onedal` respectively.

## Validation Patterns

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
- `pytest --verbose daal4py` - Native oneDAL API tests
- `pytest --verbose sklearnex` - sklearn compatibility tests
- `pytest --verbose onedal` - Low-level backend tests
- `pytest --verbose .ci/scripts/test_global_patch.py` - Global patching validation

Coverage reporting:
- `pytest --cov=onedal --cov=sklearnex --cov-config=.coveragerc --cov-branch`

### MPI/Distributed (SPMD) Testing

**Requirements:** MPI installation (Intel MPI or MPICH), mpi4py, NO_DIST!=1

**Setup MPI:**
```bash
# Option 1: Via conda (recommended)
conda install -c conda-forge impi-devel impi_rt mpi4py mpi=*=impi

# Option 2: Via pip (Intel MPI)
pip install impi-rt mpi4py --index-url https://software.repos.intel.com/python/pypi

# See doc/sources/distributed-mode.rst for detailed Intel MPI installation instructions
```

**Rebuild with MPI support:**
```bash
# Ensure NO_DIST is not set
unset NO_DIST
# First build the C++ oneDAL backend (includes SPMD libraries)
python setup.py develop
```

**Run distributed tests:**
```bash
# sklearnex SPMD tests (requires additional pytest-mpi dependency, plus dpctl and dpnp as sklearnex spmd are on GPU)
mpirun -n 4 python -m pytest --pyargs sklearnex.spmd --with-mpi

# daal4py SPMD examples
mpirun -n 4 python tests/helper_mpi_tests.py pytest --verbose -s tests/test_daal4py_spmd_examples.py
```

**Validates:** DBSCAN, K-Means, PCA, Linear Regression, Covariance, Random Forest (distributed algorithms)

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

**Configuration**: `deselected_tests.yaml` - See this file for current test list, counts, and specific reasons.

**Why Tests Are Deselected** (high-level categories):

1. **Intentional Implementation Differences**
   - Performance-optimized algorithms producing mathematically equivalent but not bit-identical results
   - Different internal algorithms achieving same outcomes (e.g., alternative solver selection)
   - Numerical precision variations from optimized computation paths

2. **Feature Coverage Gaps**
   - sklearn features not yet implemented in oneDAL backend
   - Emerging sklearn APIs (new versions) not yet supported

3. **Platform/Environment Constraints**
   - Operating system specific behaviors
   - Build toolchain differences (compiler-specific results)

4. **Test Infrastructure Differences**
   - Exception message wording differences (same error conditions, different text)
   - Edge case handling variations (rare scenarios handled differently but correctly)

**Version-Conditional Deselection**: Tests can be deselected for specific sklearn version ranges using comparison operators in `deselected_tests.yaml`.

**Impact**: Deselected tests represent a small subset of sklearn's test suite. The vast majority of algorithms work identically to sklearn. Check `deselected_tests.yaml` for specifics and see inline comments for each deselection reason.

### Running Compatibility Tests
Tests are configured in .circleci/
```bash
# Standard run
python .circleci/run_xpu_tests.py -q -d cpu --reduced --deselected_yml_file deselected_tests.yaml

# Run on GPU by specifying device and adding gpu deselections
python .circleci/run_xpu_tests.py -q -d gpu --reduced --gpu --deselected_yml_file deselected_tests.yaml

# Run tests on stock scikit-learn to compare results
python .circleci/run_xpu_tests.py -q --no-intel-optimized -d cpu --reduced --deselected_yml_file deselected_tests.yaml

## Key Testing Patterns
- Configure timeouts based on algorithm complexity (default 170s, complex up to 480s)
- Handle missing dependencies with `pytest.skip()`
- Test both sklearn compatibility and numerical accuracy
- Validate model conversion maintains prediction accuracy
- MPI tests require `mpirun -n 4` and proper MPI setup
- Check hardware availability before GPU tests
- Coverage reporting recommended for development validation

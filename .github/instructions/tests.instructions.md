# tests/* - Testing Infrastructure

## Test Structure
- `tests/`: Legacy daal4py tests and examples
- Individual module tests in respective directories (sklearnex/tests/, onedal/tests/, etc.)
- `deselected_tests.yaml`: Tests skipped in CI due to platform/dependency issues

## Test Execution Order (CRITICAL)

**Preparation**:
```bash
pip install -r requirements-test.txt
```

**Core Test Suites** (run in order):
```bash
pytest --verbose -s tests/                    # Legacy daal4py tests
pytest --verbose --pyargs daal4py             # Native oneDAL API tests
pytest --verbose --pyargs sklearnex           # sklearn compatibility tests
```

**Specific Categories**:
```bash
pytest tests/test_daal4py_examples.py         # Native API examples
pytest tests/test_model_builders.py           # XGBoost/LightGBM conversion
pytest tests/test_daal4py_spmd_examples.py    # Distributed computing (requires MPI)
```

## Test Configuration
```bash
# Environment for testing
export COVERAGE_RCFILE=$(readlink -f .coveragerc)  # Coverage configuration
export NO_DIST=1                              # Disable distributed tests
export NO_DPC=1                               # Disable GPU tests

# Memory-intensive tests may require >8GB RAM
# GPU tests require Intel GPU + drivers
# Distributed tests require MPI setup (mpirun -n 2 pytest ...)
```

## Test Categories

**Core Functionality:**
- `test_daal4py_examples.py`: Native oneDAL algorithm usage
- `test_estimators.py`: Algorithm parameter validation
- `test_printing.py`: Output formatting and verbose mode

**Compatibility:**
- `test_examples_sklearnex.py`: sklearn compatibility validation
- `test_npy.py`: NumPy array handling

**Advanced Features:**
- `test_model_builders.py`: External model conversion (XGBoost/LightGBM/CatBoost)
- `test_daal4py_serialization.py`: Model save/load functionality
- `test_daal4py_spmd_examples.py`: Distributed computing with MPI

## Deselected Tests
Tests in `deselected_tests.yaml` are skipped in CI due to:
- Platform-specific issues (Windows/Linux differences)
- Hardware requirements (GPU, specific CPU features)
- External dependencies (MPI, specific library versions)
- Memory constraints (large dataset tests)

## Development Testing
```bash
# Quick development tests (subset)
pytest tests/test_estimators.py               # Parameter validation
pytest sklearnex/tests/test_patching.py       # Core patching

# Memory/performance tests
pytest --maxfail=1 tests/                     # Stop on first failure

# Coverage testing
pytest --cov=sklearnex --cov=daal4py --cov=onedal
```

## Related Instructions
- `general.instructions.md` - Repository setup and core testing commands
- `sklearnex.instructions.md` - Testing sklearn compatibility layer
- `daal4py.instructions.md` - Testing native oneDAL algorithms
- `onedal.instructions.md` - Testing low-level bindings
- `src.instructions.md` - Testing C++/Cython core and distributed features

See individual module AGENTS.md files for module-specific testing details.
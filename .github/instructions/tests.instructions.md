# tests/* - Testing Infrastructure

## Purpose
Comprehensive validation infrastructure ensuring numerical accuracy, performance compliance, and cross-platform reliability.

## Test Structure
- `tests/` - Legacy daal4py tests and examples
- Module-specific tests in respective directories (`sklearnex/tests/`, `onedal/tests/`, `daal4py/sklearn/tests/`)
- `deselected_tests.yaml` - Tests skipped in CI due to platform/dependency issues

## Core Test Suites
```bash
pytest --verbose -s tests/
pytest --verbose daal4py
pytest --verbose sklearnex
pytest --verbose onedal
```

## Distributed (SPMD) Testing
Requirements: MPI installation (Intel MPI or MPICH), mpi4py, NO_DIST!=1. For GPU spmd tests in sklearnex/spmd pytest-mpi is also required.
```bash
mpirun -n 4 python -m pytest --pyargs sklearnex.spmd --with-mpi
mpirun -n 4 python tests/helper_mpi_tests.py pytest --verbose -s tests/test_daal4py_spmd_examples.py
```

## scikit-learn Compatibility

sklearnex tests against sklearn's test suite for API compatibility. Some tests are deselected (see `deselected_tests.yaml`) due to intentional implementation differences, unsupported features, or platform-specific issues.

For detailed coverage information, see tests/AGENTS.md.

## For GitHub Copilot

See [tests/AGENTS.md](../tests/AGENTS.md) for comprehensive information including:
- Validation patterns and numerical accuracy requirements
- Performance testing and timeout configurations
- Cross-platform testing strategies
- MPI/SPMD testing setup details

## Related Instructions
- `general.instructions.md` - Repository setup
- `sklearnex.instructions.md` - Testing sklearn compatibility
- `daal4py.instructions.md` - Testing native algorithms
- `onedal.instructions.md` - Testing low-level bindings
- `src.instructions.md` - Testing C++/Cython core and distributed features

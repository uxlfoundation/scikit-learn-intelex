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
pytest --verbose --pyargs daal4py
pytest --verbose --pyargs sklearnex
pytest --verbose --pyargs onedal
```

## Distributed (SPMD) Testing
Requirements: MPI installation (Intel MPI or OpenMPI), mpi4py, NO_DIST!=1
```bash
mpirun -n 4 python tests/helper_mpi_tests.py pytest -k spmd --with-mpi --verbose --pyargs sklearnex
mpirun -n 4 python tests/helper_mpi_tests.py pytest --verbose -s tests/test_daal4py_spmd_examples.py
```

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

# General Repository Instructions - Intel Extension for Scikit-learn

## Repository Overview

**Intel Extension for Scikit-learn** (scikit-learn-intelex) accelerates scikit-learn by 10-100x using Intel oneDAL. Zero code changes required for existing sklearn applications.

- **Languages**: Python (70%), C++ (25%), Cython (5%)
- **Architecture**: 4-layer system (sklearnex → daal4py → onedal → Intel oneDAL C++)
- **Platforms**: Linux, Windows, macOS; CPU (x86_64, ARM), GPU (Intel via SYCL)
- **Python**: 3.9-3.13 supported

## Critical Build Requirements (ALWAYS REQUIRED)

```bash
# Environment variables (MANDATORY)
export DALROOT=/path/to/onedal       # Required by setup.py:53-59
export MPIROOT=/path/to/mpi          # For distributed support

# Build dependencies (INSTALL FIRST)
pip install -r dependencies-dev     # Cython==3.1.1, numpy>=2.0, pybind11==2.13.6

# Development build (RECOMMENDED)
python setup.py develop             # Creates editable install
```

## Testing & Validation (Run in Order)

```bash
# 1. Install test dependencies
pip install -r requirements-test.txt

# 2. Core test suites
pytest --verbose -s tests/                    # Legacy tests
pytest --verbose --pyargs daal4py             # Native oneDAL tests
pytest --verbose --pyargs sklearnex           # sklearn compatibility

# 3. Code quality (REQUIRED before commit)
pre-commit install
pre-commit run --all-files --show-diff-on-failure
```

## Code Standards

- **Python**: Black (line-length=90) + isort
- **C++**: clang-format version ≥14
- **Commits**: Must be signed-off (`git commit -s`)
- **Documentation**: numpydoc format

## Common Issues & Solutions

```bash
# Build failures
export NO_DIST=1                    # Disable distributed mode if MPI issues
export NO_DPC=1                     # Disable GPU if driver issues
python setup.py build_ext --inplace --force --abs-rpath  # Linux linking

# Import/path issues
export PYTHONPATH=$(pwd)            # Add repo to path
python setup.py develop             # Ensure editable install
```

For module-specific details, see the corresponding AGENTS.md files in each directory.
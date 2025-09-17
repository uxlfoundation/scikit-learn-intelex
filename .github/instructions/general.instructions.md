# General Repository Instructions - Intel Extension for Scikit-learn

## Repository Overview

**Intel Extension for Scikit-learn** (scikit-learn-intelex) accelerates scikit-learn by 10-100x using Intel oneDAL. Zero code changes required for existing sklearn applications.

- **Languages**: Python (70%), C++ (25%), Cython (5%)
- **Architecture**: 4-layer system (sklearnex → daal4py → onedal → Intel oneDAL C++)
- **Platforms**: Linux, Windows, macOS; CPU (x86_64, ARM), GPU (Intel via SYCL)
- **Python**: 3.9-3.13 supported

## Quick Start

**Build Setup**: See [build-config.instructions.md](build-config.instructions.md) for complete details.
```bash
export DALROOT=/path/to/onedal
python setup.py develop
```

**Testing**: See [tests.instructions.md](tests.instructions.md) for comprehensive testing.
```bash
pytest --verbose --pyargs sklearnex
```

**Code Quality**:
```bash
pre-commit run --all-files
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

## Related Instructions
- `sklearnex.instructions.md` - Primary sklearn interface and patching
- `daal4py.instructions.md` - Direct oneDAL bindings and model builders
- `onedal.instructions.md` - Low-level C++ bindings
- `src.instructions.md` - Core C++/Cython implementation
- `tests.instructions.md` - Testing infrastructure and validation
- `build-config.instructions.md` - Build system and environment setup

For detailed implementation guides, see the corresponding AGENTS.md files in each directory.
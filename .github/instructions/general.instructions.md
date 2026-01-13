# General Repository Instructions - Intel Extension for Scikit-learn

## Repository Overview

Extension for Scikit-learn accelerates scikit-learn by 10-100X using oneDAL (varies by algorithm and data). Zero code changes required for existing sklearn applications.

**Architecture**: 4-layer system (sklearnex ⇒ {daal4py, onedal} → oneDAL C++)
**Platforms**: Linux, Windows; CPU (x86_64, ARM), GPU (Intel via SYCL)

**Version Requirements**:
For current supported versions, always refer to:
- `setup.py` - Python version classifiers
- `requirements-test.txt` - scikit-learn and runtime dependencies
- `dependencies-dev` - Build dependencies

## Quick Build and Test

```bash
export DALROOT=/path/to/onedal
python setup.py develop
pytest --verbose sklearnex
```

## For GitHub Copilot

See [AGENTS.md](../AGENTS.md) for comprehensive information including:
- Detailed architecture and layer interactions
- Algorithm support and GPU compatibility
- Performance patterns and optimization strategies
- Development setup and environment configuration
- Testing strategy and validation approaches

## Related Instructions
- `build-config.instructions.md` - Build system and environment setup
- `sklearnex.instructions.md` - Primary sklearn interface
- `daal4py.instructions.md` - Direct oneDAL bindings
- `onedal.instructions.md` - Low-level C++ bindings
- `src.instructions.md` - Core C++/Cython implementation
- `tests.instructions.md` - Testing infrastructure

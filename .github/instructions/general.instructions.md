# General Repository Instructions - Intel Extension for Scikit-learn

## Repository Overview

Intel Extension for Scikit-learn accelerates scikit-learn by 10-100X using Intel oneDAL (varies by algorithm and data). Zero code changes required for existing sklearn applications.

**Architecture**: 4-layer system (sklearnex → daal4py → onedal → Intel oneDAL C++)
**Platforms**: Linux, Windows; CPU (x86_64, ARM), GPU (Intel via SYCL)
**Python**: 3.9+ (officially supported versions, see setup.py classifiers)
**scikit-learn**: 1.0+ (last 4 releases + sklearn 1.0, see requirements-test.txt)

For exact dependency versions, refer to `dependencies-dev` and `requirements-test.txt` as documentation may lag updates.

## Quick Build and Test

```bash
export DALROOT=/path/to/onedal
python setup.py develop
pytest --verbose --pyargs sklearnex
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

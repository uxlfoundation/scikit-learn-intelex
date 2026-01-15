# AGENTS.md - CI/CD Infrastructure (.ci/)

## Purpose
CI/CD infrastructure for building, testing, and releasing Extension for Scikit-learn across multiple platforms.

## Key Files
- `.ci/pipeline/ci.yml` - Main CI orchestrator
- `.ci/pipeline/build-and-test-*.yml` - Platform-specific builds
- `.ci/pipeline/linting.yml` - Code quality enforcement
- `.ci/scripts/` - Automation utilities

## Platform Support
- **Linux**: Uses conda, Intel DPC++ compiler, MPI support
- **Windows**: Visual Studio 2022, conda-forge packages
- **GPU**: Intel GPU support via DPC++/SYCL

## Quality Gates
- **Linting**: black, isort, clang-format, numpydoc validation
- **Testing**: pytest with cross-platform compatibility
- **Coverage**: codecov integration

## Build Dependencies
See `dependencies-dev` and `requirements-test.txt` for exact versions.

- **oneDAL**: Downloads nightly builds from upstream oneDAL repo
- **Python**: See `setup.py` for supported versions
- **sklearn**: See `doc/sources/quick-start.rst` for supported versions
- **GPU Libraries**: dpctl, dpnp, torch for Intel GPU acceleration and Array API support

## Release Process
- **Automated**: Dynamic matrix generation for PyPI/conda releases
- **Multi-channel**: Both PyPI wheels and conda packages
- **Quality**: Automated sklearn compatibility testing before release

## Environment Variables
Key variables for development (from setup.py and CI scripts):
- `DALROOT` - Path to oneDAL (required)
- `MPIROOT` - Path to MPI for distributed computing
- `NO_DPC` - Disable GPU support
- `NO_DIST` - Disable distributed computing
- `MAKEFLAGS` - Control parallel build

## CI/CD Guidelines
- Follow established build templates in .ci/pipeline/
- Respect quality gates before submitting changes
- Use platform-specific configurations appropriately
- Test across supported Python/sklearn version combinations
- Use conda environments and virtual environments to avoid dependency conflicts

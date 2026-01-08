# Build Configuration Files

## Purpose
Build system configuration for Intel Extension for Scikit-learn using setup.py, conda, and environment variables.

## Core Build Files
- `setup.py` - Main build script
- `pyproject.toml` - Python project metadata and linting configuration
- `dependencies-dev` - Build-time dependencies
- `requirements-test.txt` - Test dependencies
- `conda-recipe/meta.yaml` - Conda package configuration

## Environment Variables
- `DALROOT` - Path to oneDAL installation (required)
- `MPIROOT` - Path to MPI for distributed features (optional)
- `NO_DIST` - Disable distributed mode
- `NO_DPC` - Disable GPU/SYCL support
- `NO_STREAM` - Disable streaming mode
- `MAKEFLAGS` - Control parallel build threads

## Build Process
1. Code Generation: oneDAL C++ headers → Python/Cython sources
2. oneDAL Bindings: cmake + pybind11 compilation
3. Cython Processing: .pyx files → C++ sources
4. Final Compilation: Link into Python extensions

## Dependencies

**Always refer to source files for exact versions:**
- `dependencies-dev` - Build dependencies (Cython, numpy, pybind11, cmake, setuptools, etc.)
- `requirements-test.txt` - Runtime and test dependencies
- `setup.py` - Python version classifiers and compatibility

**Key requirements:**
- Python 3.9+ (see `setup.py` for supported versions)
- Intel oneDAL (see `requirements-test.txt` for version)
- scikit-learn (see `requirements-test.txt` for version)

## Build Commands
- `python setup.py develop` - Development mode (editable install)
- `python setup.py install` - Production install
- `python setup.py build_ext --inplace --force` - Extensions only

## For GitHub Copilot

See [.ci/AGENTS.md](../.ci/AGENTS.md) for comprehensive information including:
- Platform-specific build configurations
- CI/CD pipeline details
- Common build issues and solutions
- Environment setup best practices

## Related Instructions
- `general.instructions.md` - Quick start commands
- `src.instructions.md` - C++/Cython build details
- `tests.instructions.md` - Testing after builds

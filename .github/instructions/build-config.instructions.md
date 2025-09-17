# Build Configuration Files

## Core Build Files
- `setup.py`: Main build script (500+ lines, complex configuration)
- `pyproject.toml`: Python project metadata + linting configuration
- `dependencies-dev`: Build-time dependencies (Cython, numpy, pybind11, cmake)
- `requirements-test.txt`: Test dependencies with version constraints
- `conda-recipe/meta.yaml`: Conda package build configuration

## Environment Variables (Critical)
```bash
# MANDATORY for building
export DALROOT=/path/to/onedal               # oneDAL installation path (required)

# OPTIONAL but commonly needed
export MPIROOT=/path/to/mpi                  # MPI for distributed features
export NO_DIST=1                             # Disable distributed mode
export NO_DPC=1                              # Disable GPU/SYCL support
export NO_STREAM=1                           # Disable streaming mode
export DEBUG_BUILD=1                         # Debug symbols + no optimization
export MAKEFLAGS=-j$(nproc)                  # Parallel build threads
```

## Build Process (4 Stages)
1. **Code Generation**: oneDAL C++ headers → Python/Cython sources
2. **oneDAL Bindings**: cmake + pybind11 compilation
3. **Cython Processing**: .pyx files → C++ sources
4. **Final Compilation**: Link everything into Python extensions

## Dependencies
**Build Dependencies (dependencies-dev):**
- Cython==3.1.1 (exact version required)
- numpy>=2.0 (version varies by Python version)
- pybind11==2.13.6
- cmake==4.0.2
- setuptools==79.0.1

**Runtime Dependencies:**
- Intel oneDAL 2021.1+ (backwards compatible)
- numpy (version-specific, see requirements-test.txt)
- scikit-learn 1.0-1.7 (see compatibility matrix)

## Build Commands
```bash
# Development build (RECOMMENDED)
python setup.py develop                       # Creates .egg-link, editable

# Production builds
python setup.py install                       # Full install
python setup.py build_ext --inplace --force   # Extensions only

# Special flags (Linux)
python setup.py build --abs-rpath             # Absolute RPATH for custom oneDAL

# Conda build
conda build .                                 # Uses conda-recipe/meta.yaml
```

## Common Build Issues
```bash
# oneDAL not found
RuntimeError: "Not set DALROOT variable"
→ Solution: export DALROOT=/path/to/onedal

# MPI required but missing
ValueError: "'MPIROOT' is not set, cannot build with distributed mode"
→ Solution: export NO_DIST=1 or set MPIROOT

# Cython version mismatch
→ Solution: pip install Cython==3.1.1 (exact version)

# Linking issues (Linux)
→ Solution: Use --abs-rpath flag
```

## CI/CD Configuration
- **GitHub Actions**: `.github/workflows/ci.yml`
- **Azure DevOps**: `.ci/pipeline/ci.yml` (main CI system)
- **Pre-commit**: `.pre-commit-config.yaml` (code quality)

Build timeouts: 120 minutes in CI (can be slow due to oneDAL compilation)
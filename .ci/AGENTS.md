# AGENTS.md - CI/CD Infrastructure (.ci/)

## Purpose
CI/CD infrastructure for building, testing, and releasing Intel Extension for Scikit-learn across multiple platforms.

## Key Files for Agents
- `.ci/pipeline/ci.yml` - Main CI orchestrator
- `.ci/pipeline/build-and-test-*.yml` - Platform-specific builds
- `.ci/pipeline/linting.yml` - Code quality enforcement
- `.ci/scripts/` - Automation utilities

## Platform Support
- **Linux/macOS**: Uses conda, Intel DPC++ compiler, MPI support
- **Windows**: Visual Studio 2022, conda-forge packages
- **GPU**: Intel GPU support via DPC++/SYCL (dpctl, dpnp packages)

## Quality Gates
- **Linting**: black, isort, clang-format, numpydoc validation
- **Testing**: pytest with cross-platform compatibility
- **Coverage**: codecov integration with threshold enforcement

## Build Dependencies
- **oneDAL**: Downloads nightly builds from upstream oneDAL repo
- **Python**: Matrix testing across Python 3.9-3.13 (verified in .ci/pipeline/ci.yml)
- **sklearn**: Multiple version compatibility (1.0-1.7)
- **GPU Libraries**: dpctl, dpnp for Intel GPU acceleration

## Release Process
- **Automated**: Dynamic matrix generation for PyPI/conda releases
- **Multi-channel**: Both PyPI wheels and conda packages
- **Quality**: Automated sklearn compatibility testing before release

## Local Development Setup

### Quality Tools Configuration (from pyproject.toml)
```bash
# Code formatting
black --line-length 90 <files>
isort --profile black --line-length 90 <files>

# C++ formatting
clang-format --style=file <cpp_files>

# Documentation validation
numpydoc-validation <python_files>
```

### Build Dependencies Download
```bash
# oneDAL nightly builds (from .github/workflows/ci.yml)
# Automatically downloads from uxlfoundation/oneDAL nightly builds
# Sets DALROOT to downloaded oneDAL location
```

### Platform-Specific Build Commands

**Linux/macOS** (from .ci/pipeline/build-and-test-lnx.yml):
```bash
# Install DPC++ compiler
bash .ci/scripts/install_dpcpp.sh

# Set up environment
source /opt/intel/oneapi/compiler/latest/env/vars.sh
export DPCPPROOT=/opt/intel/oneapi/compiler/latest

# Create conda environment
conda create -q -y -n CB -c conda-forge python=3.11 mpich pyyaml
conda activate CB
pip install -r dependencies-dev

# Build
./conda-recipe/build.sh
```

**Windows** (from .ci/pipeline/build-and-test-win.yml):
```batch
# Visual Studio setup
call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall" x64

# Build
call conda-recipe\bld.bat
```

### Environment Variables for Development
```bash
# From setup.py and CI scripts
export DALROOT=/path/to/onedal          # Required
export DPCPPROOT=/opt/intel/oneapi/compiler/latest  # For GPU support
export MPIROOT=/path/to/mpi             # For distributed computing
export NO_DPC=1                         # Disable GPU support
export NO_DIST=1                        # Disable distributed computing
export SKLEARNEX_VERSION=2024.7.0       # Version override
export MAKEFLAGS="-j$(nproc)"           # Parallel build
```

## For AI Agents
- Follow established build templates
- Respect quality gates (linting, testing, coverage)
- Use platform-specific configurations appropriately
- Test across supported Python/sklearn version combinations
- Set required environment variables (DALROOT, DPCPPROOT, MPIROOT)
- Use conda environments to avoid dependency conflicts
- Run pre-commit hooks before submitting changes
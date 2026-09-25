# AGENTS.md - Extension for scikit-learn

Accelerates scikit-learn with oneDAL. Python, C++ (pybind11) and Cython; CPU (x86_64, ARM) and Intel GPU via SYCL.

## Architecture
```text
sklearnex/  ->  daal4py/ (Cython)  -> oneDAL C++
            ->  onedal/  (pybind11) -> oneDAL C++
```
- `sklearnex/`: scikit-learn API, patching, dispatch between oneDAL and scikit-learn
- `daal4py/`: Cython bindings generated from oneDAL headers, model builders
- `onedal/`: pybind11 bindings, data conversion, CPU/GPU backend selection
- `src/`: C++/Cython core shared by both bindings

## Key Files
- `sklearnex/dispatcher.py`: patching and algorithm dispatch
- `sklearnex/_device_offload.py`: device selection and offloading
- `sklearnex/_config.py`: `config_context` options
- `onedal/__init__.py`: backend selection (DPC++/host)

## Version Floors
The floors are enforced in code; read them there rather than trusting prose:
- oneDAL: `ONEDAL_VERSION` check in `setup.py` (currently >= 2025.0)
- scikit-learn and Python: `install_requires` / `python_requires` in `setup.py`
- `sklearn_check_version` gates below the scikit-learn floor are always true and should be deleted
- Test and build dependencies: `requirements-test.txt`, `dependencies-dev`

## Code Generation
`generator/` generates daal4py's Cython bindings from oneDAL C++ headers. Modify `generator/wrappers.py` to add new oneDAL algorithms; use direct Python implementation for sklearn compatibility layers.

Never edit the generated `build/daal4py_cy.pyx`; change `generator/` and rebuild.

## Build & Test
```bash
export DALROOT=/path/to/onedal   # required; setup.py fails with "Not set DALROOT variable"
pip install -r dependencies-dev
python setup.py develop

pytest sklearnex/linear_model/tests/     # one module; start here
pytest --pyargs sklearnex                # one package, as conda-recipe/run_test.sh does
```
- Build switches read by `setup.py`: `NO_DIST=1` (no MPI; otherwise `MPIROOT` must be set), `NO_DPC=1` (no GPU), `NO_STREAM=1`. More variants: `doc/sources/building-from-source.rst`.
- `conda-recipe/run_test.sh` is the full suite (legacy `tests/`, `daal4py`, `sklearnex`, `onedal`, global patching, then MPI). It is slow; run a module first.
- GPU cases come from the `get_queues()` / `get_dataframes_and_queues()` parametrizations, which only emit a GPU queue when one is available; there is no `gpu` marker.
- MPI tests need `mpirun` and `--with-mpi`; see the MPI block in `conda-recipe/run_test.sh`.

## Rules for Changes
These come from recurring review comments; each one has been asked for on several PRs.
- Comments describe the code as it will be once merged. Don't reference discarded approaches, narrate the change, or mention "this PR".
- Keep comments short and plain: explain why in one line when one line is enough, and in two sentences rather than a paragraph. Agent-written comments have historically been bloated and hard to read; don't restate the code, hedge, or add emphasis.
- One PR, one logical change. Drive-by fixes, renames, and mechanical changes (formatting, codegen, mass renames) go in their own PRs.
- Search before adding a helper, fixture, constant table, or validation routine. Extend the existing one, and name it in the PR description.
- Don't add a lock, guard, `try`/`except`, or redundant check unless you can name the failure it prevents.
- A bug fix comes with a test that fails without the fix.
- Don't hardcode versions, URLs, or paths that a source-of-truth file or Renovate already tracks.
- New functions get full type hints and a numpydoc docstring.
- New source files use the header `Copyright contributors to the oneDAL project`; leave existing headers alone.

## Directory Guides
Read the `AGENTS.md` nearest the files you change:
- `sklearnex/AGENTS.md`: API patterns, device offloading
- `daal4py/AGENTS.md`: Native oneDAL bindings, model builders
- `onedal/AGENTS.md`: Pybind11 implementation, memory management
- `onedal/datatypes/AGENTS.md`: Data conversion, Python C-API reference ownership
- `src/AGENTS.md`: C++/Cython core, distributed computing
- `examples/AGENTS.md`: Usage patterns and example scripts
- `tests/AGENTS.md`: Testing infrastructure, validation patterns
- `.ci/AGENTS.md`: CI/CD pipeline and build infrastructure
- `doc/AGENTS.md`: Documentation build system

# scikit-learn-intelex

See AGENTS.md for architecture overview. Subdirectories (sklearnex/, daal4py/, onedal/, src/, .ci/, tests/, examples/, doc/) each have their own AGENTS.md with component-specific context.

## Build & Test

- Requires `DALROOT` env var pointing to oneDAL installation
- Build: `python setup.py develop`
- Fast test subset: `pytest sklearnex/tests/test_<module>.py`
- Full suite takes ~40 min — don't run without asking
- GPU tests require dpctl; skip with `pytest -m "not gpu"`
- Distributed (SPMD) tests require MPI — don't run without setup confirmation

## Code Conventions

- sklearnex estimators must mirror the sklearn API exactly
- Fallback to sklearn must always be preserved (oneDAL → sklearn → error)
- Don't modify files under `generator/` — they produce generated code in `daal4py/`
- Don't edit `.ci/` workflow files without confirmation

## Key Patterns

- New estimators in sklearnex must implement `_onedal_supported()` for dispatch
- Device offloading uses `config_context(target_offload=...)` — don't bypass
- Zero-copy requires C-contiguous arrays; document when copying is unavoidable

# Copilot instructions - Extension for scikit-learn

Repository guidance lives in `AGENTS.md` files next to the code they govern. Before reviewing or editing a file, read the root `AGENTS.md` and the nearest `AGENTS.md` above the file:

| Path | Guidance |
| --- | --- |
| `sklearnex/` | `sklearnex/AGENTS.md` |
| `onedal/` | `onedal/AGENTS.md`; `onedal/datatypes/AGENTS.md` for data conversion |
| `daal4py/`, `generator/` | `daal4py/AGENTS.md` |
| `src/` | `src/AGENTS.md` |
| `tests/`, `**/tests/`, `deselected_tests.yaml` | `tests/AGENTS.md` |
| `doc/` | `doc/AGENTS.md` |
| `.ci/`, `.github/workflows/`, `conda-recipe/` | `.ci/AGENTS.md` |

## Reviewing pull requests
- Check the diff against the "Rules for Changes" sections of the relevant `AGENTS.md` files. Those rules come from recurring maintainer review comments.
- Report at most 10 findings, most severe first. Correctness, reference leaks, host copies of device data, and missing test coverage outrank style.
- Don't report what pre-commit or CI already enforces: black, isort, clang-format, numpydoc validation, codespell, and license headers.
- Only report a finding you have confirmed in the source. Say so when a finding depends on code outside the diff that you couldn't read.
- Flag PR descriptions that don't match the diff, and PRs that bundle unrelated changes.
- Keep comments short and concrete; include a suggestion block when the fix is small.

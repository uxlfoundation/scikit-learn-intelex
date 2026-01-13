# sklearnex/* - Primary sklearn-compatible Interface

## Purpose
Primary user interface for sklearn acceleration with patching system and device offloading.

## Key Files
- `dispatcher.py` - Patching system
- `_device_offload.py` - GPU/CPU dispatch
- `_config.py` - Global configuration
- `base.py` - oneDALEstimator base class

## Usage Patterns
- Global Patching: `patch_sklearn()` accelerates all supported sklearn imports
- Selective Patching: `patch_sklearn(["DBSCAN", "KMeans"])` for specific algorithms
- Direct Import: `from sklearnex.cluster import DBSCAN` without patching
- Device Control: `config_context(target_offload="gpu:0")` for GPU execution

## Testing
```bash
pytest --verbose sklearnex
pytest sklearnex/tests/test_patching.py
pytest sklearnex/tests/test_config.py
```

## For GitHub Copilot

See [sklearnex/AGENTS.md](../sklearnex/AGENTS.md) for comprehensive information including:
- Detailed patching system architecture
- Device offloading and fallback mechanisms
- Algorithm support conditions and GPU compatibility
- Configuration API and context management

## Related Instructions
- `general.instructions.md` - Repository setup
- `onedal.instructions.md` - Low-level backend
- `tests.instructions.md` - Testing compatibility layer

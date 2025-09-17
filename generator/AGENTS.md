# AGENTS.md - Code Generator (generator/)

## Purpose
Automated code generation system that creates Python bindings for oneDAL algorithms through C++ header parsing and Jinja2 templates.

## Key Files
- `gen_daal4py.py` - Main orchestrator (1274 lines)
- `parse.py` - C++ header parser (727 lines)
- `wrapper_gen.py` - Jinja2 template engine (1626 lines)
- `wrappers.py` - Algorithm metadata configuration (1028 lines)
- `format.py` - Type conversion utilities (287 lines)

## Generation Pipeline
1. **Header Parsing**: Extract classes, enums, templates from oneDAL C++ headers
2. **Metadata Processing**: Filter algorithms, handle required parameters
3. **Template Generation**: Create Cython wrappers using Jinja2 templates
4. **Code Output**: Generate Python API with proper type conversion

## Algorithm Configuration
```python
# Required parameters for algorithms
required = {
    "algorithms::dbscan": [("epsilon", "fptype"), ("minObservations", "size_t")],
    "algorithms::kmeans": [("nClusters", "size_t"), ("maxIterations", "size_t")],
    # ... 40+ algorithm configurations
}
```

## Template System
- **Jinja2 Templates**: Generate consistent Cython wrappers
- **Type Mapping**: Python ↔ C++ type conversion
- **Error Handling**: Input validation and exception handling
- **Memory Management**: Proper C++ object lifecycle

## When to Modify Generator vs Python Code

### Modify Generator (`wrappers.py`) When:
```python
# Adding new oneDAL algorithms not yet wrapped
required = {
    "algorithms::new_algorithm": [("param1", "size_t"), ("param2", "double")]
}

# Changing algorithm parameter requirements
no_constructor = {
    "algorithms::special_case": {"param": ["type", "default_value"]}
}
```

### Direct Python Implementation When:
- Adding sklearn interface compatibility layers
- Implementing parameter validation and conversion
- Creating custom error handling or fallback logic
- Adding utility functions that don't require C++ bindings

### Build Process Integration
```bash
# Generator runs in stage 1 of 4-stage build (from INSTALL.md)
# 1. Creating C++ and Cython sources from oneDAL C++ headers
# 2. Building oneDAL Python interfaces via cmake and pybind11
# 3. Running Cython on generated sources
# 4. Compiling and linking them

# Force regeneration during development
python setup.py build_ext --inplace --force
```

### Debugging Generated Code
- Generated files appear in build directories
- Check `generated_sources/` for Cython output
- Use `print()` statements in `wrapper_gen.py` templates for debugging
- Template variables available: `{{ns}}`, `{{algo}}`, `{{args_decl}}`, etc.

## For AI Agents
- Generator runs automatically during build
- Modify `wrappers.py` to add new algorithm configurations
- Templates in `wrapper_gen.py` handle code patterns
- Type mappings in `format.py` for new data types
- Test generation changes with `python setup.py build_ext --inplace --force`
- Use direct Python implementation for sklearn compatibility layers
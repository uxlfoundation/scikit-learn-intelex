# AGENTS.md - Documentation (doc/)

## Purpose
Sphinx-based documentation generation system for Intel Extension for Scikit-learn.

## Key Files
- `sources/conf.py`: Sphinx configuration with extensions
- `build-doc.sh`: Documentation build automation
- `sources/algorithms.rst`: Algorithm support matrix
- `sources/daal4py.rst`: API reference with autodoc

## Build System
- **Sphinx Extensions**: autodoc, nbsphinx, intersphinx, napoleon
- **Notebook Integration**: Jupyter notebooks included via nbsphinx
- **Cross-References**: Links to sklearn, numpy, pandas documentation
- **Deployment**: Automated GitHub Pages deployment on releases

## Content Structure
- **User Guides**: Quick start, installation, performance optimization
- **API Reference**: Auto-generated from docstrings
- **Examples**: Real-world applications and Jupyter tutorials
- **Developer Docs**: Distributed computing, contribution guidelines

## Build Commands
Local development: `make html`
Production deployment: `./build-doc.sh --gh-pages`

## Documentation Guidelines
- Use reStructuredText format for documentation files
- Include proper docstrings for autodoc generation
- Test documentation builds locally before submitting
- Maintain cross-references and intersphinx links

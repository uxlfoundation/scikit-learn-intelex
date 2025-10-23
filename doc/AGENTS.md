# AGENTS.md - Documentation (doc/)

## Purpose
Sphinx-based documentation generation system for Intel Extension for Scikit-learn.

## Key Files for Agents
- `doc/sources/conf.py` - Sphinx configuration with extensions
- `doc/build-doc.sh` - Documentation build automation
- `doc/sources/algorithms.rst` - Algorithm support matrix
- `doc/sources/daal4py.rst` - API reference with autodoc

## Build System
- **Sphinx Extensions**: autodoc, nbsphinx, intersphinx, napoleon
- **Notebook Integration**: Jupyter notebooks included via nbsphinx
- **Cross-References**: Links to sklearn, numpy, pandas documentation
- **GitHub Pages**: Automated deployment on releases

## Content Structure
- **User Guides**: Quick start, performance optimization
- **API Reference**: Auto-generated from docstrings
- **Examples**: Real-world applications (kaggle/, notebooks/)
- **Developer Docs**: Distributed computing, contribution guidelines

## Build Commands
```bash
# Local development
make html

# Production deployment
./build-doc.sh --gh-pages
```

## For AI Agents
- Use reStructuredText format for documentation
- Include proper docstrings for autodoc generation
- Test documentation builds locally before submitting
- Maintain cross-references and intersphinx links
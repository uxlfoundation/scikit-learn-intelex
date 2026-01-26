# AGENTS.md - Examples (examples/)

## Purpose
Comprehensive Python scripts and Jupyter notebooks demonstrating Intel Extension for Scikit-learn usage patterns.

## Directory Structure
- `daal4py/`: Native oneDAL API examples
- `sklearnex/`: Accelerated sklearn examples, specifically spmd and online algorithm use
- `mb/`: Model builder examples (XGBoost/LightGBM/CatBoost conversion)
- `notebooks/`: Jupyter tutorials with real datasets
- `utils/`: Utility functions

## Key Usage Patterns

### Native oneDAL API
Direct algorithm access via daal4py for maximum performance and control.

### Accelerated sklearn
Global patching or selective imports to accelerate sklearn code transparently.

### GPU Acceleration
Device offloading examples using config_context for supported algorithms.

### Distributed Computing (SPMD)
MPI-based distributed execution with daalinit/daalfini coordination.

### Model Conversion
Converting XGBoost/LightGBM/CatBoost models to oneDAL format for accelerated inference.

## Algorithm Categories
- **Clustering**: DBSCAN, K-Means
- **Linear Models**: Linear, Ridge, Logistic regression, Lasso, ElasticNet
- **Ensemble**: Random Forest, Gradient Boosting
- **Decomposition**: PCA, SVD, Incremental PCA
- **Statistics**: Moments, covariance, correlation
- **SVM**: Classification and regression
- **Neighbors**: KNN classification and regression

## Usage Guidelines
- Use examples as templates for new implementations
- Follow established patterns for performance optimization
- Include both sklearn and oneDAL performance comparisons
- Test examples across CPU/GPU configurations

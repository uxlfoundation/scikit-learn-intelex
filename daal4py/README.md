<!--
  ~ Copyright 2021 Intel Corporation
  ~
  ~ Licensed under the Apache License, Version 2.0 (the "License");
  ~ you may not use this file except in compliance with the License.
  ~ You may obtain a copy of the License at
  ~
  ~     http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing, software
  ~ distributed under the License is distributed on an "AS IS" BASIS,
  ~ WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  ~ See the License for the specific language governing permissions and
  ~ limitations under the License.
-->

# daal4py - A Convenient Python API to the oneAPI Data Analytics Library
[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=main)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=main)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/21716/badge.svg)](https://scan.coverity.com/projects/daal4py)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/IntelPython/daal4py/discussions)
[![PyPI Version](https://img.shields.io/pypi/v/daal4py)](https://pypi.org/project/daal4py/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/daal4py)](https://anaconda.org/conda-forge/daal4py)

**IMPORTANT NOTICE**: `daal4py` has been merged into `scikit-learn-intelex`. As of version 2025.0, it is distributed as an additional importable module within the package `scikit-learn-intelex` instead of being a separate package. The last standalone release of `daal4py` was version 2024.7, and this standalone package will not receive further updates.

A simplified API to oneAPI Data Analytics Library that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.  Built to help provide an abstraction to oneAPI Data Analytics Library for either direct usage or integration into one's own framework.

Note: For the most part, `daal4py` is used as an internal backend within the Scikit-Learn extension, and it is highly recommended to use `sklearnex` instead. Nevertheless, some functionalities from `daal4py` can still be of use, and the module can still be imported directly (`import daal4py`) after installing `scikit-learn-intelex`.

## 👀 Follow us on Medium

We publish blogs on Medium, so [follow us](https://medium.com/intel-analytics-software/tagged/machine-learning) to learn tips and tricks for more efficient data analysis the help of daal4py. Here are our latest blogs:

- [Intel Gives Scikit-Learn the Performance Boost Data Scientists Need](https://medium.com/intel-analytics-software/intel-gives-scikit-learn-the-performance-boost-data-scientists-need-42eb47c80b18)
- [From Hours to Minutes: 600x Faster SVM](https://medium.com/intel-analytics-software/from-hours-to-minutes-600x-faster-svm-647f904c31ae)
- [Improve the Performance of XGBoost and LightGBM Inference](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Kaggle Challenges Using Intel AI Analytics Toolkit](https://medium.com/intel-analytics-software/accelerate-kaggle-challenges-using-intel-ai-analytics-toolkit-beb148f66d5a)
- [Accelerate Your scikit-learn Applications](https://medium.com/intel-analytics-software/improving-the-performance-of-xgboost-and-lightgbm-inference-3b542c03447e)
- [Accelerate Linear Models for Machine Learning](https://medium.com/intel-analytics-software/accelerating-linear-models-for-machine-learning-5a75ff50a0fe)
- [Accelerate K-Means Clustering](https://medium.com/intel-analytics-software/accelerate-k-means-clustering-6385088788a1)

## 🔗 Important links
- [Documentation](https://intelpython.github.io/daal4py/)
- [scikit-learn API and patching](https://intelpython.github.io/daal4py/sklearn.html#sklearn)
- [Building from Sources](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/daal4py/INSTALL.md)
- [About oneAPI Data Analytics Library](https://github.com/uxlfoundation/oneDAL)

## 💬 Support

Report issues, ask questions, and provide suggestions using:

- [GitHub Issues](https://github.com/uxlfoundation/scikit-learn-intelex/issues)
- [GitHub Discussions](https://github.com/uxlfoundation/scikit-learn-intelex/discussions)
- [Forum](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)

You may reach out to project maintainers privately at onedal.maintainers@intel.com

# 🛠 Installation

Daal4Py is distributed as part of scikit-learn-intelex, which itself is distributed under different channels.

See the [installation instructions for scikit-learn-intelex](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md) for details.

⚠️ Note: *GPU and MPI support are optional dependencies.
Required dependencies for GPU and MPI support will not be downloaded.
You need to manually install `dpcpp_cpp_rt` and `dpctl` packages for GPU support, and `mpi4py` with `impi_rt` as backend package for MPI support.*

<details><summary>[Click to expand] ℹ️ How to install dpcpp_cpp_rt and impi_rt packages </summary>

```shell
# PyPi for dpcpp
pip install -U dpcpp_cpp_rt dpctl
```

```shell
# PyPi for MPI
pip install -U mpi4py impi_rt
```

```shell
# conda for dpcpp
conda install dpcpp_cpp_rt dpctl -c https://software.repos.intel.com/python/conda/
```

```shell
# conda for MPI
conda install mpi4py impi_rt -c https://software.repos.intel.com/python/conda/
```

</details>


# ⚠️ Scikit-learn patching

Scikit-learn patching functionality in daal4py was deprecated and moved to a separate package - [Extension for Scikit-learn*](https://github.com/uxlfoundation/scikit-learn-intelex). All future updates for the patching will be available in Extension for Scikit-learn only. Please use the package instead of daal4py for the Scikit-learn acceleration.

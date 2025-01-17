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

# daal4py - A Convenient Python API to the Intel(R) oneAPI Data Analytics Library
[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=main)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=main)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/21716/badge.svg)](https://scan.coverity.com/projects/daal4py)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/IntelPython/daal4py/discussions)
[![PyPI Version](https://img.shields.io/pypi/v/daal4py)](https://pypi.org/project/daal4py/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/daal4py)](https://anaconda.org/conda-forge/daal4py)


A simplified API to Intel(R) oneAPI Data Analytics Library that allows for fast usage of the framework suited for Data Scientists or Machine Learning users.  Built to help provide an abstraction to Intel(R) oneAPI Data Analytics Library for either direct usage or integration into one's own framework.

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
- [About Intel(R) oneAPI Data Analytics Library](https://github.com/uxlfoundation/oneDAL)

## 💬 Support

Report issues, ask questions, and provide suggestions using:

- [GitHub Issues](https://github.com/uxlfoundation/scikit-learn-intelex/issues)
- [GitHub Discussions](https://github.com/uxlfoundation/scikit-learn-intelex/discussions)
- [Forum](https://community.intel.com/t5/Intel-Distribution-for-Python/bd-p/distribution-python)

You may reach out to project maintainers privately at onedal.maintainers@intel.com

# 🛠 Installation
daal4py is available at the [Python Package Index](https://pypi.org/project/daal4py/),
on Anaconda Cloud in [Conda-Forge channel](https://anaconda.org/conda-forge/daal4py)
and in [Intel channel](https://anaconda.org/intel/daal4py).

```bash
# PyPI (recommended by default)
pip install daal4py
```

```bash
# Anaconda Cloud from Conda-Forge channel (recommended for conda users by default)
conda install daal4py -c conda-forge
```

```bash
# Intel channel (recommended for Intel® Distribution for Python users)
conda install daal4py -c https://software.repos.intel.com/python/conda/
```

⚠️ Note: *GPU and MPI support are optional dependencies.
Required dependencies for GPU and MPI support will not be downloaded.
You need to manually install ***dpcpp_cpp_rt*** package for GPU support and ***impi_rt*** package for MPI support.*

<details><summary>[Click to expand] ℹ️ How to install dpcpp_cpp_rt and impi_rt packages </summary>

```bash
# PyPi for dpcpp
pip install --upgrade dpcpp_cpp_rt
```

```bash
# PyPi for MPI
pip install --upgrade impi_rt
```

```bash
# Anaconda Cloud for dpcpp
conda install dpcpp_cpp_rt -c intel
```

```bash
# Anaconda Cloud for MPI
conda install impi_rt -c intel
```

<details><summary>[Click to expand] ℹ️ Supported configurations </summary>

#### 📦 PyPi channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :-------------:| :------------:| :------------:|
|    **Linux**            |    [CPU, GPU]  |  [CPU, GPU]    |   [CPU, GPU]  |  [CPU, GPU]|  |
|    **Windows**          |    [CPU, GPU]  |  [CPU, GPU]    |   [CPU, GPU]  |  [CPU, GPU]|  |
|    **OsX**              |    [CPU]       |  [CPU]         |    [CPU]      |    [CPU]      |

#### 📦 Anaconda Cloud: Conda-Forge channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :------------: | :------------:| :------------:|
|    **Linux**            |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **Windows**          |   [CPU]        |   [CPU]        |     [CPU]     |     [CPU]     |
|    **OsX**              |   ❌           |     ❌        |     ❌        |       ❌     |

#### 📦 Anaconda Cloud: Intel channel

| OS / Python version     | **Python 3.6** | **Python 3.7** | **Python 3.8**| **Python 3.9**|
| :-----------------------| :------------: | :-------------:| :------------:| :------------:|
|    **Linux**            |   ❌          |     [CPU, GPU]  |     ❌       |      ❌       |
|    **Windows**          |   ❌          |     [CPU, GPU]  |     ❌       |      ❌       |
|    **OsX**              |   ❌          |     [CPU]       |     ❌       |      ❌       |

</details>

You can [build daal4py from sources](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md) as well.


# ⚠️ Scikit-learn patching

Scikit-learn patching functionality in daal4py was deprecated and moved to a separate package - [Intel(R) Extension for Scikit-learn*](https://github.com/uxlfoundation/scikit-learn-intelex). All future updates for the patching will be available in Intel(R) Extension for Scikit-learn only. Please use the package instead of daal4py for the Scikit-learn acceleration.

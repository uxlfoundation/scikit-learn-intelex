<!--
  ~ Copyright 2018 Intel Corporation
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

<div align="center">


# Extension for Scikit-learn*

<h3> Speed up your scikit-learn applications for CPUs and GPUs across single- and multi-node configurations

[Releases](https://github.com/uxlfoundation/scikit-learn-intelex/releases)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Documentation](https://uxlfoundation.github.io/scikit-learn-intelex/)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Examples](https://github.com/uxlfoundation/scikit-learn-intelex/tree/master/examples/notebooks)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;[Support](SUPPORT.md)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;[License](https://github.com/uxlfoundation/scikit-learn-intelex/blob/master/LICENSE)&nbsp;&nbsp;&nbsp;


[![Build Status](https://dev.azure.com/daal/daal4py/_apis/build/status/CI?branchName=main)](https://dev.azure.com/daal/daal4py/_build/latest?definitionId=9&branchName=main)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/21716/badge.svg)](https://scan.coverity.com/projects/daal4py)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/uxlfoundation/scikit-learn-intelex/badge)](https://securityscorecards.dev/viewer/?uri=github.com/uxlfoundation/scikit-learn-intelex)
[![Join the community on GitHub Discussions](https://badgen.net/badge/join%20the%20discussion/on%20github/black?icon=github)](https://github.com/uxlfoundation/scikit-learn-intelex/discussions)
[![PyPI Version](https://img.shields.io/pypi/v/scikit-learn-intelex)](https://pypi.org/project/scikit-learn-intelex/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/scikit-learn-intelex)](https://anaconda.org/conda-forge/scikit-learn-intelex)
[![python version](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![scikit-learn supported versions](https://img.shields.io/badge/sklearn-1.0%20%7C%201.2%20%7C%201.3%20%7C%201.4%20%7C%201.5%20%7C%201.6%20%7C%201.7-blue)](https://img.shields.io/badge/sklearn-1.0%20%7C%201.2%20%7C%201.3%20%7C%201.4%20%7C%201.5%20%7C%201.6%20%7C%201.7-blue)

---
</h3>

<div align="left">

## Overview

Extension for Scikit-learn is a **free software AI accelerator** designed to deliver over **10-100X** acceleration to your existing scikit-learn code.
The software acceleration is achieved with vector instructions, AI hardware-specific memory optimizations, threading, and optimizations.


With Extension for Scikit-learn, you can:

* Speed up training and inference by up to 100x with equivalent mathematical accuracy
* Benefit from performance improvements across different CPU hardware configurations, including GPUs and multi-GPU configurations
* Integrate the extension into your existing Scikit-learn applications without code modifications
* Continue to use the open-source scikit-learn API
* Enable and disable the extension with a couple of lines of code or at the command line

## Acceleration

![](https://raw.githubusercontent.com/uxlfoundation/scikit-learn-intelex/master/doc/sources/_static/scikit-learn-acceleration.PNG)

[Benchmarks code](https://github.com/IntelPython/scikit-learn_bench)

## Optimizations

Easiest way to benefit from accelerations from the extension is by patching scikit-learn with it:

- **Enable CPU optimizations**

    ```python
    import numpy as np
    from sklearnex import patch_sklearn
    patch_sklearn()

    from sklearn.cluster import DBSCAN

    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                  [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    ```

- **Enable GPU optimizations**

    _Note: executing on GPU has [additional system software requirements](https://www.intel.com/content/www/us/en/developer/articles/system-requirements/intel-oneapi-dpcpp-system-requirements.html) - see [details](https://uxlfoundation.github.io/scikit-learn-intelex/latest/oneapi-gpu.html)._

    ```python
    import numpy as np
    from sklearnex import patch_sklearn, config_context
    patch_sklearn()

    from sklearn.cluster import DBSCAN

    X = np.array([[1., 2.], [2., 2.], [2., 3.],
                  [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
    with config_context(target_offload="gpu:0"):
        clustering = DBSCAN(eps=3, min_samples=2).fit(X)
    ```

:eyes: Check out available [notebooks](https://github.com/uxlfoundation/scikit-learn-intelex/tree/master/examples/notebooks) for more examples.

### Usage without patching

Alternatively, all functionalities are also available under a separate module which can be imported directly, without involving any patching.

* To run on CPU:

  ```python
  import numpy as np
  from sklearnex.cluster import DBSCAN

  X = np.array([[1., 2.], [2., 2.], [2., 3.],
                [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
  clustering = DBSCAN(eps=3, min_samples=2).fit(X)
  ```

* To run on GPU:

  ```python
  import numpy as np
  from sklearnex import config_context
  from sklearnex.cluster import DBSCAN

  X = np.array([[1., 2.], [2., 2.], [2., 3.],
                [8., 7.], [8., 8.], [25., 80.]], dtype=np.float32)
  with config_context(target_offload="gpu:0"):
      clustering = DBSCAN(eps=3, min_samples=2).fit(X)
  ```

## Installation

To install Extension for Scikit-learn, run:

```shell
pip install scikit-learn-intelex
```

Package is also offered through other channels such as conda-forge. See all installation instructions in the [Installation Guide](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/INSTALL.md).

## Integration

The easiest way of accelerating scikit-learn workflows with the extension is through through [patching](https://uxlfoundation.github.io/scikit-learn-intelex/latest/quick-start.html#patching), which replaces the stock scikit-learn algorithms with their optimized versions provided by the extension using the same namespaces in the same modules as scikit-learn.

The patching only affects [supported algorithms and their parameters](https://uxlfoundation.github.io/scikit-learn-intelex/latest/algorithms.html).
You can still use not supported ones in your code, the package simply fallbacks into the stock version of scikit-learn.

> **_TIP:_** Enable [verbose mode](https://uxlfoundation.github.io/scikit-learn-intelex/latest/verbose.html) to see which implementation of the algorithm is currently used.

To patch scikit-learn, you can:
* Use the following command-line flag:
  ```shell
  python -m sklearnex my_application.py
  ```
* Add the following lines to the script:
  ```python
  from sklearnex import patch_sklearn
  patch_sklearn()
  ```

:eyes: Read about [other ways to patch scikit-learn](https://uxlfoundation.github.io/scikit-learn-intelex/latest/quick-start.html#patching).

As an alternative, accelerated classes from the extension can also be imported directly without patching, thereby allowing to keep them separate from stock scikit-learn ones - for example:

```python
from sklearnex.cluster import DBSCAN as exDBSCAN
from sklearn.cluster import DBSCAN as stockDBSCAN

# ...
```

## Documentation

* [Quick Start](https://uxlfoundation.github.io/scikit-learn-intelex/latest/quick-start.html)
* [Documentation and Tutorials](https://uxlfoundation.github.io/scikit-learn-intelex/latest/index.html)
* [Release Notes](https://github.com/uxlfoundation/scikit-learn-intelex/releases)
* [Medium Blogs](https://uxlfoundation.github.io/scikit-learn-intelex/latest/blogs.html)
* [Code of Conduct](https://github.com/uxlfoundation/scikit-learn-intelex/blob/master/CODE_OF_CONDUCT.md)

### Extension and oneDAL

Acceleration in patched scikit-learn classes is achieved by replacing calls to scikit-learn with calls to oneDAL (oneAPI Data Analytics Library) behind the scenes:
- [oneAPI Data Analytics Library](https://github.com/uxlfoundation/oneDAL)

## Samples & Examples

* [Examples](https://github.com/uxlfoundation/scikit-learn-intelex/tree/master/examples/notebooks)
* [Samples](https://uxlfoundation.github.io/scikit-learn-intelex/latest/samples.html)
* [Kaggle Kernels](https://uxlfoundation.github.io/scikit-learn-intelex/latest/kaggle.html)


## How to Contribute

We welcome community contributions, check our [Contributing Guidelines](https://github.com/uxlfoundation/scikit-learn-intelex/blob/master/CONTRIBUTING.md) to learn more.

------------------------------------------------------------------------
\* The Intel logo, and other Intel marks are trademarks of Intel Corporation or its subsidiaries. Other names and brands may be claimed as the property of others.


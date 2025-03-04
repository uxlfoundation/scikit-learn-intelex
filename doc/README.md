<!-- file: README.md
******************************************************************************
* Copyright 2024 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# Build Documentation

Our documentation is written in restructured text markup and built with [Sphinx](http://www.sphinx-doc.org/en/master/).

## Generate Documentation

To build Intel(R) Extension for Scikit-Learn documentation locally:

1. Clone the repository:

		git clone https://github.com/uxlfoundation/scikit-learn-intelex.git

2. Install the `scikit-learn-intelex` package.

3. Install required documentation builder dependencies using `pip`:

		pip install -r requirements-doc.txt

4. Go to the `doc` folder:

		cd scikit-learn-intelex/doc

5. Run the ``build-doc.sh`` script. 

You will then find documentation under the `_build/html` folder.

Additional documentation for the `daal4py` module can be built with the following steps (after having installed the `scikit-learn-intelex` packages and doc dependencies):

1. Go to the `doc/daal4py` folder:

		cd scikit-learn-intelex/doc/daal4py

2. Run the makefile (assuming linux):

		make html

Documentation will similarly be built under `_build/html`.

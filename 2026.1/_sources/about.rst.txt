.. Copyright contributors to the oneDAL project
..
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..     http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. include:: substitutions.rst

=====================
About the |sklearnex|
=====================

The |sklearnex| is a free and open-source software accelerator built atop of the |sklearn| and the :external+onedal:doc:`oneDAL <index>` (|onedal|) libraries.

It mostly works by replacing selected calls to algorithms in |sklearn| with calls to the |onedal| library, which offers more optimized versions of the same routines (see :doc:`algorithms`). The optimizations in the |onedal| in turn are achieved by leveraging SIMD instructions and exploiting cache structures of modern hardware, along with using the `oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`__ library for linear algebra operations in place of the `OpenBLAS <https://www.openmathlib.org/OpenBLAS/>`__ library used by default by |sklearn|.

Unlike other libraries in the Python ecosystem, classes and functions in the |sklearnex| are not just :external+sklearn:doc:`scikit-learn-compatible <developers/develop>`, but rather are built atop of |sklearn| itself by inheriting from their classes directly, defining the same attributes that the stock version of |sklearn| would do for each estimator, and reusing most of scikit-learn's estimator methods where appropriate.

The |sklearnex| is regularly tested for API compatibility and for correctness against |sklearn|'s own test suite (see :ref:`conformance_tests` for more information), and can be easily swapped in place of the stock |sklearn| library by :doc:`patching <patching>` it.

The |sklearnex| aims to be compatible with the last 3 minor releases of |sklearnex| available at any given time, in addition to the 1.0 release as a special case, and ensures this compatibility by offering different code routes according to the |sklearn| version encountered at runtime - for example, if a given attribute of a class is removed in version 1.x of |sklearn|, the |sklearnex| will not set that attribute when running with |sklearn| >=1.x, but will still do so when running with |sklearn| <1.x, in order to guarantee full API compatibility.

Performance of the |sklearnex| is regularly measured and compared against that of other libraries using public and synthetic datasets through `sklbench <https://github.com/IntelPython/scikit-learn_bench>`__, which is also free and fully open-source.

Initially developed by Intel as the Intel Extension for Scikit-learn*, the |sklearnex| and the |onedal| are now projects under the `UXL Foundation <https://uxlfoundation.org>`__ umbrella, and can be built from source to provide accelerated routines for other platforms such as ARM and RISCV - see :doc:`building-from-source` for more information.

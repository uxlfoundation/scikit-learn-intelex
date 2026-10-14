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

================================
FAQ (Frequently Asked Questions)
================================

What is the relationship between |sklearn| and the |sklearnex|?
---------------------------------------------------------------

The |sklearnex| is a separate library providing classes and functions which build on top of their |sklearn| analogs (i.e. an extension), with some methods being overridden to replace calls to |sklearn| internals with calls to more optimized equivalents from the |onedal| while still producing the same results, up to numerical accuracy.

These optimized versions are under an importable module named ``sklearnex`` which provides an API-compatible alternative to ``sklearn`` for a selected subset of classes and functions (see :doc:`supported algorithms <algorithms>`). Given the API compatibility, it is possible to :doc:`monkeypatch <patching>` the ``sklearn`` module to automatically use the |sklearnex| whenever possible without further code changes.

Is ``sklearnex`` a drop-in replacement for ``sklearn``?
-------------------------------------------------------

Not entirely, since ``sklearnex`` offers only a subset of the classes and functions from ``sklearn``, but when the same classes and functions are offered (such as ``sklearnex.linear_model.LinearRegression``), they do act as drop-in replacements with the exact same API and behaviors.

However, when applying :doc:`monkeypatching <patching>`, the modified ``sklearn`` module does become a drop-in replacement for stock |sklearn|.

Do I need a specific version of |sklearn| with the |sklearnex|?
---------------------------------------------------------------

The current release of the |sklearnex| aims to be compatible with the last 4 minor releases of |sklearn| available at any given time, typically covering a period of approximately 2 years of releases. See the `README <https://github.com/intel/scikit-learn-intelex>`__ badges for more information about supported versions at any given moment.

How does the |sklearnex| ensure API compatibility with different versions of |sklearn|?
---------------------------------------------------------------------------------------

Classes from the |sklearnex| inherit from their analog classes in |sklearn|, exposing the same set of public attributes and methods, and reusing many functionalities from |sklearn| without modifications, oftentimes calling code from |sklearn| itself to fill in gaps in features between the |onedal| and |sklearn|.

Runtime version checks performed on the ``sklearn`` module enable dispatching of different code branches within the |sklearnex| to ensure matching behavior against multiple possible versions, ensuring compatibility in version-dependent details such as renamed attributes, user warnings, and similar.

The |sklearnex| is regularly tested for both API compatibility and for correctness against |sklearn|'s own test suite (see :ref:`conformance_tests` for more information) to verify this.

Does the |sklearnex| always produce the exact same results as |sklearn|?
------------------------------------------------------------------------

When accelerated routines from the |sklearnex| are used, results (e.g. numbers returned by ``.predict()``, coefficients fitted by :obj:`sklearn.linear_model.LinearRegression`, etc.) can be expected to match against what would be obtained through |sklearn| up to numerical precision for the floating point type (``float32`` / ``float64``) supplied as input data (note that non-floating point types are casted to ``float64``), but outputs might not necessarily be exactly equal byte-by-byte.

When it comes to procedures that employ random numbers (such as :func:`sklearn.model_selection.train_test_split`), results might differ a bit since the |sklearnex| will generate different sequences of random numbers, but they should match in expectation over large samples.

Additionally, some algorithms might have methodological differences that yield slightly different results - for instance, forest ensembles (such as :obj:`sklearn.ensemble.RandomForestClassifier`) use the histogram method (i.e. features are binned) for calculating splits, which might not always produce the same result on average as the sorted-indices method used by |sklearn|, but should result in observing very similar numbers in cross-validated metrics such as :obj:`~sklearn.metrics.roc_auc_score` or :obj:`~sklearn.metrics.r2_score`.

What makes the |sklearnex| faster?
----------------------------------

The |sklearnex| replaces selected calls to internal functions and methods in |sklearn| with more optimized equivalents from the |onedal|, which is a C++ library offering high-performance implementations of common machine learning routines. These optimizations in turn come from leveraging SIMD instructions and exploiting cache structures of modern hardware, along with using the `oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`__ library for linear algebra operations in place of the `OpenBLAS <https://www.openmathlib.org/OpenBLAS/>`__ library used by default by |sklearn|.

See the list of :doc:`supported algorithms <algorithms>` for more information about what is accelerated by the |sklearnex| and what isn't.

Who develops the |sklearnex|?
-----------------------------

The |sklearnex| is developed by the `UXL Foundation <https://uxlfoundation.org>`__, a consortium for open multi-vendor standards in accelerated computing.

Does patching modify files from |sklearn|?
------------------------------------------

Not by default. The scope of :func:`~sklearnex.patch_sklearn` is limited to the Python process in which it was called, and will not persist across sessions.

However, patching can optionally be applied permanently by modifying an existing |sklearn| installation with ``patch_sklearn(global_patch=True)`` - see :doc:`patching` for more details. This permanent patching in turn can be undone by ``unpatch_sklearn(global_patch=True)``.

Can I use the library without patching |sklearn|?
-------------------------------------------------

Yes, it is possible to import estimators and functions from the ``sklearnex`` module without doing any modifications to the ``sklearn`` module. See :doc:`Quick Start <index>` for usage examples with and without patching.

Will all workflows run faster under patching?
---------------------------------------------

No. The |sklearnex| only accelerates a selected subset of routines from |sklearn|. See :doc:`algorithms` for details.

What happens when some routine is not supported?
------------------------------------------------

When an operation is not supported by the |sklearnex|, the original code branches from |sklearn| will be called instead, which is referred throughout these docs as a 'fallback'. Therefore, results are the same as what would be obtained when not using the |sklearnex|.

In these cases however, there will be no acceleration. The underlying library being used for a given computation can be verified using :doc:`verbose mode <verbose>`. See :doc:`algorithms` for more information about what is covered by the |sklearnex|.

Should I use the |sklearnex| if I already use |sklearn| with :external+numpy:doc:`NumPy <user/whatisnumpy>` and SciPy `backed by oneMKL <https://github.com/intel/optimization-zone/blob/main/software/numpy/README.md>`__?
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Yes - beyond usage of accelerated routines from `oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`__, the |sklearnex| introduces other forms of optimizations that cannot be replicated by swapping BLAS or LAPACK implementations, such as different parallelization strategies for calculations of covariances and Gram matrices, or usage of more specialized linear algebra factorizations than the ones used by default NumPy solvers whenever applicable.

Should I use the |sklearnex| if I already use array API with Torch or DPNP?
---------------------------------------------------------------------------

Yes - for both CPU and GPU, the |sklearnex| is able to leverage finer-grained optimizations that are not feasible when using the limited subset of functionalities mandated by the array API standard. See other points throughout this doc page for more info.

How is the |sklearnex| faster than |sklearn| on GPU if both use array API?
--------------------------------------------------------------------------

Compared to |sklearn|, when `array API inputs <https://data-apis.org/array-api/>`__ (such as `Torch Tensors <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>`__ or |dpnp_array|) are passed to the |sklearnex|, most computations will not be handled through the array's namespace object the way |sklearn| would do.

Instead, memory pointers, devices, and queues are extracted from the array objects and forwarded to the |onedal| for the most computationally-heavy operations, which in turn employs more specialized compute kernels defined through the SYCL framework for some operations, and makes calls to the `DPC routines <https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/current/overview.html>`__ from `oneMKL <https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html>`__ whenever applicable.

See :doc:`array_api` for more information.

Why is array API not working when I try it?
-------------------------------------------

In order to use array API, it must first :external+sklearn:doc:`be enabled <modules/array_api>` in both |sklearn| and SciPy, and operations must be run under a :doc:`config_context <config-contexts>` or global settings must be modified for it to work.

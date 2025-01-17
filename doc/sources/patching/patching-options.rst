.. Copyright 2021 Intel Corporation
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

.. important::

    These patching methods are interchangeable.
    They support different enabling scenarios while producing the same result.

- Without editing the code of a scikit-learn application by using the following command line flag::

    python -m sklearnex my_application.py

- Directly from the script::

    from sklearnex import patch_sklearn
    patch_sklearn()

.. important::

    You have to import scikit-learn **after** these lines. Otherwise, the patching will not affect the original scikit-learn estimators.

- Through importing the desired estimator from the sklearnex module in your script::

    from sklearnex.neighbors import NearestNeighbors

- Through :ref:`global patching <global_patching>` to enable patching for your scikit-learn installation for all further runs::

    python -m sklearnex.glob patch_sklearn

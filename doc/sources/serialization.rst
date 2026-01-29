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

==============================
Model serialization (pickling)
==============================

Serializing objects
-------------------

Objects in Python are bound to the process that creates them. Usually, when it comes to statistical or machine learning models, one typically wants to save fitted models for later usage - for example, by fitting a model on a large machine, saving it to disk storage, and then serving it (making predictions on new data) on other smaller machines.

Just like other objects in Python, estimator objects from the |sklearnex| can be serialized / persisted / pickled using the built-in ``pickle`` module - for example:

.. code-block:: python

    import pickle
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearnex.linear_model import LinearRegression

    X, y = make_regression()
    model = LinearRegression().fit(X, y)

    model_file = "linear_model.pkl"
    with open(model_file, "wb") as output_file:
        pickle.dump(model, output_file)

    with open(model_file, "rb") as input_file:
        model_deserialized = pickle.load(input_file)

    np.testing.assert_array_equal(
        model_deserialized.predict(X),
        model.predict(X),
    )

.. hint:: Note that, while operations performed on CPU are usually deterministic and procedures involving random numbers allow controlling the seed, upon deserializing a model in a different machine, it is not guaranteed that outputs such as predictions on new data will be byte-by-byte reproducible due to differences in instructions sets supported by different CPUs, runtimes of backend libraries, and similar such nuances. Nevertheless, results from predictions of the same model on different machines with compatible environments (see next section) should be within numerical roundoff error.

Serialization requirements
--------------------------

All estimator classes in the |sklearnex| that have a counterpart in |sklearn| (and thus participate in :ref:`patching <patching>`) inherit from that respective class from |sklearn|, and expose the same public attributes. Hence, in order to successfully serialize and deserialize a model from the |sklearnex|, it is necessary to satisfy all the requirements for serialization of |sklearn| objects, such as using the same |sklearn| version for serializing and deserializing the object - see :ref:`sklearn:pickle_persistence` for more details.

In addition to those requirements, additional conditions need to be met in order to ensure that serialization and deserialization of objects belonging to classes from the |sklearnex| will work correctly:

- The versions of both |sklearn| and the |sklearnex| must be the same for deserializing a given object as the versions used for serializing it.
- The version of the :external+onedal:doc:`oneDAL <index>` backend used for the |sklearnex| (through Python package ``dal`` or ``daal`` depending on the installation medium) must be either the same or a higher minor version within the same major version series - for example, |onedal| version 2025.10 can deserialize models saved with 2025.8, but not the other way around, and version 2026.0 might not be able to deserialize models from 2025.x versions.
- Other dependencies providing data classes that constitute object attributes, such as NumPy's arrays, must also be able to successfully serialize and deserialize in that same environment. Note that :ref:`array API classes <array_api>`, which might be used as object attributes when enabling this mode, might have tighter serialization requirements than NumPy.
- The Python major version must be the same, and the minor version must be either the same or higher.

Just like in |sklearn|, in order to ensure that deserialized models work correctly, it is highly recommended to recreate the same environment that created the serialized model in terms of python versions, package versions, and configurations of packages (e.g. build variants in the case of conda-managed environments).

.. warning:: Note that, unlike objects from |sklearn|, objects from the |sklearnex| will not necessarily issue a warning when deserializing them with an incompatible library version.

Serialization of GPU models
---------------------------

Be aware that if using the :ref:`target offload option <target_offload>` to fit models on GPU or on another SYCL device, upon deserialization of those models, the internal data behind them will be re-created on host (CPU), hence the deserialized models will become CPU/host ones and will not be able to make predictions on GPU data.

If persistence of GPU-only models is desired, one can instead use :ref:`array API classes with GPU support <array_api>`, which might have a different logic for serialization that preserves the device.

Currently, the only array API library with SYCL support known to provide serializable GPU arrays is `PyTorch <https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html>`__.

.. warning:: If serialization of models is desired, avoid usage of |dpnp| GPU arrays as they are not serilizable.

Example:

.. code-block:: python

    import os
    os.environ["SCIPY_ARRAY_API"] = "1"

    import pickle
    import torch
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearnex import config_context
    from sklearnex.linear_model import LinearRegression

    X_np, y_np = make_regression()
    X = torch.tensor(X_np, dtype=torch.float32, device="xpu")
    y = torch.tensor(y_np, dtype=torch.float32, device="xpu")

    with config_context(array_api_dispatch=True):
        model = LinearRegression().fit(X, y)
        pred_fresh = model.predict(X)

    assert isinstance(pred_fresh, torch.Tensor)

    model_deserialized = pickle.loads( pickle.dumps(model) )
    with config_context(array_api_dispatch=True):
        pred_deserialized = model_deserialized.predict(X)

    np.testing.assert_allclose(
        pred_fresh.cpu().numpy(),
        pred_deserialized.cpu().numpy(),
    )

Configurations are not serializable
-----------------------------------

Be aware that serialization of model objects does not imply saving of global or local configurations. For example, a model that was fitted to :ref:`array API classes <array_api>` will have those same array API classes as attributes, but array API mode is not enabled by default in |sklearn| (and by extension, not in the |sklearnex| either). Hence, if the :ref:`global configuration <config_contexts>` was modified to enable array API support, the deserialized model might not be usable in a new Python process until that setting (array API) is enabled.

Likewise, other process-level internal settings, such as efficiency parameters that are modifiable through static class methods of estimators (currently undocumented), are not saved along with a model object, since they are not managed by it.

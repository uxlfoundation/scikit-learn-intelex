# ===============================================================================
# Copyright 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================

import array_api_strict
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import is_regressor
from sklearn.datasets import make_classification, make_regression

from daal4py.sklearn._utils import (
    _package_check_version,
    daal_check_version,
    sklearn_check_version,
)
from onedal.tests.utils._dataframes_support import (
    _as_numpy,
    _convert_to_dataframe,
    dpnp_available,
    get_dataframes_and_queues,
    torch_available,
    torch_xpu_available,
)
from onedal.tests.utils._device_selection import is_sycl_device_available

if dpnp_available:
    import dpnp
if torch_available:
    import torch

hparam_values = [
    (None, None, None, None),
    (8, 100, 32, 0.3),
    (16, 100, 32, 0.3),
    (32, 100, 32, 0.3),
    (64, 10, 32, 0.1),
    (128, 100, 1000, 1.0),
]


@pytest.fixture
def hyperparameters(request):
    from sklearnex.ensemble import RandomForestClassifier

    hparams = RandomForestClassifier.get_hyperparameters("predict")

    def restore_hyperparameters():
        RandomForestClassifier.reset_hyperparameters("predict")

    request.addfinalizer(restore_hyperparameters)
    return hparams


@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
@pytest.mark.parametrize("block, trees, rows, scale", hparam_values)
def test_sklearnex_import_rf_classifier(
    hyperparameters, dataframe, queue, block, trees, rows, scale
):
    from sklearnex.ensemble import RandomForestClassifier

    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    rf = RandomForestClassifier(max_depth=2, random_state=0).fit(X, y)

    if hyperparameters and block is not None:
        hyperparameters.block_size = block
        hyperparameters.min_trees_for_threading = trees
        hyperparameters.min_number_of_rows_for_vect_seq_compute = rows
        hyperparameters.scale_factor_for_vect_parallel_compute = scale
    assert "sklearnex" in rf.__module__
    X_test = _convert_to_dataframe([[0, 0, 0, 0]], sycl_queue=queue, target_df=dataframe)
    assert_allclose([1], _as_numpy(rf.predict(X_test)))


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_rf_regression(dataframe, queue):
    if (not daal_check_version((2025, "P", 200))) and queue and queue.sycl_device.is_gpu:
        pytest.skip("Skipping due to bug in histogram merges fixed in 2025.2.")
    from sklearnex.ensemble import RandomForestRegressor

    X, y = make_regression(n_features=4, n_informative=2, random_state=0, shuffle=False)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    rf = RandomForestRegressor(max_depth=2, random_state=0).fit(X, y)
    assert "sklearnex" in rf.__module__
    X_test = _convert_to_dataframe([[0, 0, 0, 0]], sycl_queue=queue, target_df=dataframe)
    pred = _as_numpy(rf.predict(X_test))

    # Check that the prediction is within a reasonable range.
    # 'y' should be in the neighborhood of zero for x=0.
    assert pred[0] >= -10
    assert pred[0] <= 10

    # Check that the trees aren't just empty nodes predicting the mean
    for estimator in rf.estimators_:
        assert estimator.tree_.children_left.shape[0] > 1


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_et_classifier(dataframe, queue):
    if (not daal_check_version((2025, "P", 200))) and queue and queue.sycl_device.is_gpu:
        pytest.skip("Skipping due to bug in histogram merges fixed in 2025.2.")
    from sklearnex.ensemble import ExtraTreesClassifier

    X, y = make_classification(
        n_samples=1000,
        n_features=4,
        n_informative=2,
        n_redundant=0,
        random_state=0,
        shuffle=False,
    )
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    # For the 2023.2 release, random_state is not supported
    # defaults to seed=777, although it is set to 0
    rf = ExtraTreesClassifier(max_depth=2, random_state=0).fit(X, y)
    assert "sklearnex" in rf.__module__
    X_test = _convert_to_dataframe([[0, 0, 0, 0]], sycl_queue=queue, target_df=dataframe)
    assert_allclose([1], _as_numpy(rf.predict(X_test)))


@pytest.mark.parametrize("dataframe,queue", get_dataframes_and_queues())
def test_sklearnex_import_et_regression(dataframe, queue):
    if (not daal_check_version((2025, "P", 200))) and queue and queue.sycl_device.is_gpu:
        pytest.skip("Skipping due to bug in histogram merges fixed in 2025.2.")
    from sklearnex.ensemble import ExtraTreesRegressor

    X, y = make_regression(n_features=1, random_state=0, shuffle=False)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)
    # For the 2023.2 release, random_state is not supported
    # defaults to seed=777, although it is set to 0
    rf = ExtraTreesRegressor(random_state=0).fit(X, y)
    assert "sklearnex" in rf.__module__
    X_test = _convert_to_dataframe([[0]], sycl_queue=queue, target_df=dataframe)
    pred = _as_numpy(rf.predict(X_test))

    # Check that the prediction is within a reasonable range.
    # 'y' should be in the neighborhood of zero for x=0.
    assert pred[0] >= -10
    assert pred[0] <= 10

    # Check that the trees aren't just empty nodes predicting the mean
    for estimator in rf.estimators_:
        assert estimator.tree_.children_left.shape[0] > 1


@pytest.mark.allow_sklearn_fallback
@pytest.mark.parametrize("dataframe, queue", get_dataframes_and_queues())
def test_classifiers_work_on_single_class(dataframe, queue):
    from sklearnex.ensemble import ExtraTreesClassifier, RandomForestClassifier

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(20, 10))
    y = np.zeros(X.shape[0])
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    y = _convert_to_dataframe(y, sycl_queue=queue, target_df=dataframe)

    np.testing.assert_array_equal(
        _as_numpy(RandomForestClassifier(n_estimators=1).fit(X, y).predict(X)),
        _as_numpy(y),
    )
    np.testing.assert_array_equal(
        _as_numpy(ExtraTreesClassifier(n_estimators=1).fit(X, y).predict(X)),
        _as_numpy(y),
    )


@pytest.mark.allow_sklearn_fallback
def test_classifiers_work_on_single_class_non_numeric():
    from sklearnex.ensemble import ExtraTreesClassifier, RandomForestClassifier

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(20, 10))
    y = pd.Series(np.repeat("qwerty", X.shape[0]))

    np.testing.assert_array_equal(
        RandomForestClassifier(n_estimators=1).fit(X, y).predict(X),
        y,
    )
    np.testing.assert_array_equal(
        ExtraTreesClassifier(n_estimators=1).fit(X, y).predict(X),
        y,
    )


# TODO: add 'sample_weights' to this test once oneDAL supports the
# new scikit-learn methodology and sklearnex doesn't fall back.
@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later scikit-learn versions.",
)
@pytest.mark.skipif(
    not _package_check_version("2.1", np.__version__),
    reason="Array API functionality requires more recent version of NumPy.",
)
@pytest.mark.parametrize("X_xp", [np, pd, array_api_strict])
@pytest.mark.parametrize("y_xp", [np, pd, array_api_strict])
@pytest.mark.parametrize("class_weight", [None, "balanced"])
@pytest.mark.parametrize("n_classes", [0, 2, 3])  # 0 == regression
def test_rf_mixed_array_namespaces(X_xp, y_xp, class_weight, n_classes, with_array_api):
    if class_weight is not None and n_classes == 0:
        pytest.skip()
    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    if n_classes == 0:  # regressor
        y = rng.standard_normal(size=X.shape[0])
    else:
        y = rng.integers(n_classes, size=X.shape[0])

    if X_xp is pd:
        X = pd.DataFrame(X)
    else:
        X = X_xp.asarray(X)
    if y_xp is pd:
        if n_classes != 0:
            y = np.array(["a", "b", "c"])[y]
        y = pd.Series(y)
    else:
        y = y_xp.asarray(y)

    from sklearnex.ensemble import RandomForestClassifier, RandomForestRegressor

    model = (RandomForestClassifier if n_classes != 0 else RandomForestRegressor)(
        n_estimators=10
    )
    if class_weight is not None:
        model.set_params(class_weight=class_weight)
    model.fit(X, y)
    pred = model.predict(X)
    _ = model.score(X, y)

    if n_classes == 0:
        assert pred.__class__ == (X.__class__ if X_xp is not pd else np.ndarray)
    else:
        if y_xp is pd:
            assert isinstance(model.classes_, np.ndarray)
        else:
            assert model.classes_.__class__ == y.__class__
        proba = model.predict_proba(X)
        if X_xp is pd:
            assert isinstance(proba, np.ndarray)
        else:
            assert proba.__class__ == X.__class__

    # Note: this is a quick check to ensure that the result has the same
    # kind of values as the input. Note that roughly 1/2 or 1/3 of the inputs
    # will be of the same class, so this ensures that different values are
    # predicted. There's no particular justification behind requiring 60%
    # classification accuracy.
    if n_classes != 0:
        if y_xp is pd:
            y_xp = np
        pred_is_correct = y_xp.astype(y_xp.asarray(pred == y), y_xp.float32)
        assert y_xp.sum(pred_is_correct) >= (0.60 * int(X.shape[0]))


@pytest.mark.skipif(
    not sklearn_check_version("1.9"),
    reason="Functionality introduced in later scikit-learn versions.",
)
@pytest.mark.skipif(
    not is_sycl_device_available("gpu"), reason="Test checks GPU-specific functionality."
)
@pytest.mark.parametrize(
    "X_xp, X_device",
    ([(torch, "xpu"), (torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "gpu"), (dpnp, "cpu")] if dpnp_available else []),
)
@pytest.mark.parametrize(
    "y_xp, y_device",
    ([(torch, "xpu"), (torch, "cpu")] if torch_xpu_available else [])
    + ([(dpnp, "gpu"), (dpnp, "cpu")] if dpnp_available else [])
    + [(pd, None)],
)
@pytest.mark.parametrize(
    "estimator_class",
    [
        "RandomForestRegressor",
        "RandomForestClassifier",
        "ExtraTreesRegressor",
        "ExtraTreesClassifier",
    ],
)
def test_rf_mixed_devices(
    X_xp, y_xp, X_device, y_device, estimator_class, with_array_api
):
    from sklearnex import ensemble

    model = getattr(ensemble, estimator_class)(n_estimators=2)

    rng = np.random.default_rng(seed=123)
    X = rng.standard_normal(size=(50, 4))
    if is_regressor(model):
        y = rng.standard_normal(size=X.shape[0])
    else:
        y = rng.integers(2, size=X.shape[0])

    X = X_xp.asarray(X, device=X_device)
    if y_xp is pd:
        if is_regressor(model):
            y = pd.Series(y)
        else:
            y = pd.Series(np.array(["a", "b"])[y])
    else:
        y = y_xp.asarray(y, device=y_device)

    model.fit(X, y)
    pred = model.predict(X)
    if is_regressor(model):
        assert pred.__class__ == X.__class__
    else:
        if y_xp is pd:
            assert isinstance(pred, np.ndarray)
        else:
            assert pred.__class__ == y.__class__
        proba = model.predict_proba(X)
        assert proba.__class__ == X.__class__
    _ = model.score(X, y)

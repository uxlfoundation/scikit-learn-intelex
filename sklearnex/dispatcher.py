# ==============================================================================
# Copyright 2021 Intel Corporation
# Copyright 2024 Fujitsu Limited
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
# ==============================================================================

import os
import sys
from functools import lru_cache
from typing import Optional, Union

from daal4py.sklearn._utils import daal_check_version, sklearn_check_version


def _is_new_patching_available():
    return os.environ.get("OFF_ONEDAL_IFACE", "0") == "0" and daal_check_version(
        (2021, "P", 300)
    )


def _is_preview_enabled():
    return os.environ.get("SKLEARNEX_PREVIEW") is not None


@lru_cache(maxsize=None)
def get_patch_map_core(preview=False):
    if preview:
        # use recursion to guarantee that state of preview
        # and non-preview maps are done at the same time.
        # The two lru_cache dicts are actually one underneath.
        # Preview is always secondary. Both sklearnex patch
        # maps are referring to the daal4py dict unless the
        # key has been replaced. Use with caution.
        mapping = get_patch_map_core().copy()

        if _is_new_patching_available():
            import sklearn.covariance as covariance_module
            import sklearn.decomposition as decomposition_module
            import sklearn.linear_model as linear_model_module

            # Preview classes for patching
            from .preview.covariance import (
                EmpiricalCovariance as EmpiricalCovariance_sklearnex,
            )
            from .preview.decomposition import IncrementalPCA as IncrementalPCA_sklearnex

            if daal_check_version((2024, "P", 1)):
                from .preview.linear_model import (
                    LogisticRegressionCV as LogisticRegressionCV_sklearnex,
                )

            # Since the state of the lru_cache without preview cannot be
            # guaranteed to not have already enabled sklearnex algorithms
            # when preview is used, setting the mapping element[1] to None
            # should NOT be done. This may lose track of the unpatched
            # sklearn estimator or function.
            # Covariance
            mapping["empiricalcovariance"] = [
                [
                    (
                        covariance_module,
                        "EmpiricalCovariance",
                        EmpiricalCovariance_sklearnex,
                    ),
                    None,
                ]
            ]

            # IncrementalPCA
            mapping["incrementalpca"] = [
                [
                    (
                        decomposition_module,
                        "IncrementalPCA",
                        IncrementalPCA_sklearnex,
                    ),
                    None,
                ]
            ]

            # LogisticRegressionCV
            if daal_check_version((2024, "P", 1)):
                mapping["logisticregressioncv"] = [
                    [
                        (
                            linear_model_module,
                            "LogisticRegressionCV",
                            LogisticRegressionCV_sklearnex,
                        ),
                        None,
                    ]
                ]
            else:
                if "logisticregressioncv" in mapping:
                    mapping.pop("logisticregressioncv")
            if "log_reg_cv" in mapping:
                mapping.pop("log_reg_cv")

        return mapping

    from daal4py.sklearn.monkeypatch.dispatcher import _get_map_of_algorithms

    # NOTE: this is a shallow copy of a dict, modification is dangerous
    mapping = _get_map_of_algorithms().copy()

    # NOTE: Use of daal4py _get_map_of_algorithms and
    # get_patch_map/get_patch_map_core should not be used concurrently.
    # The setting of elements to None below may cause loss of state
    # when interacting with sklearn. A dictionary key must not be
    # modified but totally replaced, otherwise it will cause chaos.
    # Hence why pop is being used.
    if _is_new_patching_available():
        # Scikit-learn* modules
        import sklearn as base_module
        import sklearn.cluster as cluster_module
        import sklearn.covariance as covariance_module
        import sklearn.decomposition as decomposition_module
        import sklearn.dummy as dummy_module
        import sklearn.ensemble as ensemble_module

        if sklearn_check_version("1.4"):
            import sklearn.ensemble._gb as _gb_module
        else:
            import sklearn.ensemble._gb_losses as _gb_module
        import sklearn.linear_model as linear_model_module
        import sklearn.manifold as manifold_module
        import sklearn.metrics as metrics_module
        import sklearn.model_selection as model_selection_module
        import sklearn.neighbors as neighbors_module
        import sklearn.svm as svm_module

        if sklearn_check_version("1.2.1"):
            import sklearn.utils.parallel as parallel_module
        else:
            import sklearn.utils.fixes as parallel_module

        # Classes and functions for patching
        from ._config import config_context as config_context_sklearnex
        from ._config import get_config as get_config_sklearnex
        from ._config import set_config as set_config_sklearnex
        from .cluster import DBSCAN as DBSCAN_sklearnex
        from .cluster import KMeans as KMeans_sklearnex
        from .covariance import (
            IncrementalEmpiricalCovariance as IncrementalEmpiricalCovariance_sklearnex,
        )
        from .decomposition import PCA as PCA_sklearnex
        from .dummy import DummyRegressor as DummyRegressor_sklearnex
        from .ensemble import ExtraTreesClassifier as ExtraTreesClassifier_sklearnex
        from .ensemble import ExtraTreesRegressor as ExtraTreesRegressor_sklearnex
        from .ensemble import RandomForestClassifier as RandomForestClassifier_sklearnex
        from .ensemble import RandomForestRegressor as RandomForestRegressor_sklearnex
        from .linear_model import ElasticNet as ElasticNet_sklearnex
        from .linear_model import (
            IncrementalLinearRegression as IncrementalLinearRegression_sklearnex,
        )
        from .linear_model import IncrementalRidge as IncrementalRidge_sklearnex
        from .linear_model import Lasso as Lasso_sklearnex
        from .linear_model import LinearRegression as LinearRegression_sklearnex
        from .linear_model import LogisticRegression as LogisticRegression_sklearnex
        from .linear_model import Ridge as Ridge_sklearnex
        from .manifold import TSNE as TSNE_sklearnex
        from .metrics import pairwise_distances as pairwise_distances_sklearnex
        from .metrics import roc_auc_score as roc_auc_score_sklearnex
        from .model_selection import train_test_split as train_test_split_sklearnex
        from .neighbors import KNeighborsClassifier as KNeighborsClassifier_sklearnex
        from .neighbors import KNeighborsRegressor as KNeighborsRegressor_sklearnex
        from .neighbors import LocalOutlierFactor as LocalOutlierFactor_sklearnex
        from .neighbors import NearestNeighbors as NearestNeighbors_sklearnex
        from .svm import SVC as SVC_sklearnex
        from .svm import SVR as SVR_sklearnex
        from .svm import NuSVC as NuSVC_sklearnex
        from .svm import NuSVR as NuSVR_sklearnex
        from .utils.parallel import _FuncWrapper as _FuncWrapper_sklearnex

        # DBSCAN
        mapping.pop("dbscan")
        mapping["dbscan"] = [[(cluster_module, "DBSCAN", DBSCAN_sklearnex), None]]

        # KMeans
        mapping.pop("kmeans")
        mapping["kmeans"] = [[(cluster_module, "KMeans", KMeans_sklearnex), None]]

        # PCA
        mapping.pop("pca")
        mapping["pca"] = [[(decomposition_module, "PCA", PCA_sklearnex), None]]

        # SVM
        mapping.pop("svm")
        mapping.pop("svc")
        mapping["svr"] = [[(svm_module, "SVR", SVR_sklearnex), None]]
        mapping["svc"] = [[(svm_module, "SVC", SVC_sklearnex), None]]
        mapping["nusvr"] = [[(svm_module, "NuSVR", NuSVR_sklearnex), None]]
        mapping["nusvc"] = [[(svm_module, "NuSVC", NuSVC_sklearnex), None]]

        # ElasticNet
        mapping.pop("elasticnet")
        mapping["elasticnet"] = [
            [
                (
                    linear_model_module,
                    "ElasticNet",
                    ElasticNet_sklearnex,
                ),
                None,
            ]
        ]

        # Lasso
        mapping.pop("lasso")
        mapping["lasso"] = [
            [
                (
                    linear_model_module,
                    "Lasso",
                    Lasso_sklearnex,
                ),
                None,
            ]
        ]

        # Linear Regression
        mapping.pop("linear")
        mapping.pop("linearregression")
        mapping["linear"] = [
            [
                (
                    linear_model_module,
                    "LinearRegression",
                    LinearRegression_sklearnex,
                ),
                None,
            ]
        ]
        mapping["linearregression"] = mapping["linear"]

        # Logistic Regression

        mapping.pop("logisticregression")
        mapping.pop("log_reg")
        mapping.pop("logistic")
        mapping.pop("_logistic_regression_path")
        mapping["log_reg"] = [
            [
                (
                    linear_model_module,
                    "LogisticRegression",
                    LogisticRegression_sklearnex,
                ),
                None,
            ]
        ]
        mapping["logisticregression"] = mapping["log_reg"]

        # This is in sklearnex preview, but daal4py doesn't have preview
        if "log_reg_cv" in mapping:
            mapping.pop("log_reg_cv")
        if "logisticregressioncv" in mapping:
            mapping.pop("logisticregressioncv")

        # Ridge
        mapping.pop("ridge")
        mapping["ridge"] = [
            [
                (
                    linear_model_module,
                    "Ridge",
                    Ridge_sklearnex,
                ),
                None,
            ]
        ]

        # manifold
        mapping.pop("tsne")
        mapping["tsne"] = [
            [
                (manifold_module, "TSNE", TSNE_sklearnex),
                None,
            ]
        ]

        # metrics
        mapping.pop("distances")
        mapping.pop("roc_auc_score")
        mapping["distances"] = [
            [
                (metrics_module, "pairwise_distances", pairwise_distances_sklearnex),
                None,
            ]
        ]
        mapping["pairwise_distances"] = mapping["distances"]
        mapping["roc_auc_score"] = [
            [
                (metrics_module, "roc_auc_score", roc_auc_score_sklearnex),
                None,
            ]
        ]

        # model_selection
        mapping.pop("train_test_split")
        mapping["train_test_split"] = [
            [
                (model_selection_module, "train_test_split", train_test_split_sklearnex),
                None,
            ]
        ]

        # kNN
        mapping.pop("knn_classifier")
        mapping.pop("kneighborsclassifier")
        mapping.pop("knn_regressor")
        mapping.pop("kneighborsregressor")
        mapping.pop("nearest_neighbors")
        mapping.pop("nearestneighbors")
        mapping["knn_classifier"] = [
            [
                (
                    neighbors_module,
                    "KNeighborsClassifier",
                    KNeighborsClassifier_sklearnex,
                ),
                None,
            ]
        ]
        mapping["knn_regressor"] = [
            [
                (
                    neighbors_module,
                    "KNeighborsRegressor",
                    KNeighborsRegressor_sklearnex,
                ),
                None,
            ]
        ]
        mapping["nearest_neighbors"] = [
            [(neighbors_module, "NearestNeighbors", NearestNeighbors_sklearnex), None]
        ]
        mapping["kneighborsclassifier"] = mapping["knn_classifier"]
        mapping["kneighborsregressor"] = mapping["knn_regressor"]
        mapping["nearestneighbors"] = mapping["nearest_neighbors"]

        # Ensemble
        mapping["extra_trees_classifier"] = [
            [
                (
                    ensemble_module,
                    "ExtraTreesClassifier",
                    ExtraTreesClassifier_sklearnex,
                ),
                None,
            ]
        ]
        mapping["extra_trees_regressor"] = [
            [
                (
                    ensemble_module,
                    "ExtraTreesRegressor",
                    ExtraTreesRegressor_sklearnex,
                ),
                None,
            ]
        ]
        mapping["extratreesclassifier"] = mapping["extra_trees_classifier"]
        mapping["extratreesregressor"] = mapping["extra_trees_regressor"]
        mapping.pop("random_forest_classifier")
        mapping.pop("random_forest_regressor")
        mapping.pop("randomforestclassifier")
        mapping.pop("randomforestregressor")
        mapping["random_forest_classifier"] = [
            [
                (
                    ensemble_module,
                    "RandomForestClassifier",
                    RandomForestClassifier_sklearnex,
                ),
                None,
            ]
        ]
        mapping["random_forest_regressor"] = [
            [
                (
                    ensemble_module,
                    "RandomForestRegressor",
                    RandomForestRegressor_sklearnex,
                ),
                None,
            ]
        ]
        mapping["randomforestclassifier"] = mapping["random_forest_classifier"]
        mapping["randomforestregressor"] = mapping["random_forest_regressor"]

        # LocalOutlierFactor
        mapping["lof"] = [
            [
                (neighbors_module, "LocalOutlierFactor", LocalOutlierFactor_sklearnex),
                None,
            ]
        ]
        mapping["localoutlierfactor"] = mapping["lof"]

        # IncrementalEmpiricalCovariance
        mapping["incrementalempiricalcovariance"] = [
            [
                (
                    covariance_module,
                    "IncrementalEmpiricalCovariance",
                    IncrementalEmpiricalCovariance_sklearnex,
                ),
                None,
            ]
        ]

        # IncrementalLinearRegression
        mapping["incrementallinearregression"] = [
            [
                (
                    linear_model_module,
                    "IncrementalLinearRegression",
                    IncrementalLinearRegression_sklearnex,
                ),
                None,
            ]
        ]

        if daal_check_version((2024, "P", 600)):
            # IncrementalRidge
            mapping["incrementalridge"] = [
                [
                    (
                        linear_model_module,
                        "IncrementalRidge",
                        IncrementalRidge_sklearnex,
                    ),
                    None,
                ]
            ]

        # DummyRegressor
        mapping["dummyregressor"] = [
            [
                (
                    dummy_module,
                    "DummyRegressor",
                    DummyRegressor_sklearnex,
                ),
                None,
            ]
        ]

        # Required patching of DummyRegressor in the gradient boosting
        # module as it is used in the GradientBoosting algorithms
        mapping["gb_dummyregressor"] = [
            [
                (
                    _gb_module,
                    "DummyRegressor",
                    DummyRegressor_sklearnex,
                ),
                None,
            ]
        ]

        # Configs
        mapping["set_config"] = [
            [(base_module, "set_config", set_config_sklearnex), None]
        ]
        mapping["get_config"] = [
            [(base_module, "get_config", get_config_sklearnex), None]
        ]
        mapping["config_context"] = [
            [(base_module, "config_context", config_context_sklearnex), None]
        ]

        # Necessary for proper work with multiple threads
        mapping["parallel.get_config"] = [
            [(parallel_module, "get_config", get_config_sklearnex), None]
        ]
        mapping["_funcwrapper"] = [
            [(parallel_module, "_FuncWrapper", _FuncWrapper_sklearnex), None]
        ]
    return mapping


# This is necessary to properly cache the patch_map when
# using preview.
def get_patch_map() -> dict[str, bool]:
    preview = _is_preview_enabled()
    return get_patch_map_core(preview=preview)


get_patch_map.cache_clear = get_patch_map_core.cache_clear


get_patch_map.cache_info = get_patch_map_core.cache_info


def get_patch_names() -> list[str]:
    return list(get_patch_map().keys())


def patch_sklearn(
    name: Optional[Union[str, list[str]]] = None,
    verbose: bool = True,
    global_patch: bool = False,
    preview: bool = False,
) -> None:
    """Apply patching to the ``sklearn`` module.

    Patches the ``sklearn`` module from |sklearn| to make calls to the accelerated
    versions of estimators and functions from the |sklearnex|, either as a whole
    or on a per-estimator basis.

    Notes
    -----
    If estimators from ``sklearn`` have already been imported before ``patch_sklearn``
    is called, they need to be re-imported in order for the patching to take effect.

    See Also
    --------
    is_patched_instance: To verify that an instance of an estimator is patched.
    unpatch_sklearn: To undo the patching.

    Parameters
    ----------
    name : str, list of str, or None
        Names of the desired estimators to patch. Can pass a single instance name (e.g.
        ``"LogisticRegression"``), or a list of names (e.g. ``["LogisticRegression", "PCA"]``).

        If ``None``, will patch all the supported estimators.

        See the :doc:`algorithm support table <algorithms>` for more information.
    verbose : bool
        Whether to print information messages about the patching being applied or not.

        Note that this refers only to a message about patching applied through this
        function. Passing ``True`` here does **not** enable :doc:`verbose mode <verbose>`
        for further estimator calls.

        When the message is printed, it will use the Python ``stderr`` stream.
    global_patch : bool
        Whether to apply the patching on the installed ``sklearn`` module itself,
        which is a mechanism that persists across sessions and processes.

        If ``True``, the ``sklearn`` module files will be modified to apply patching
        immediately upon import of this module, so that next time, importing of
        ``sklearnex`` will not be necessary.
    preview : bool
        Whether to include the :doc:`preview estimators <preview>` in the patching.

        Note that this will forcibly set the environment variable ``SKLEARNEX_PREVIEW``.

        If environment variable ``SKLEARNEX_PREVIEW`` is set at the moment this function
        is called, preview estimators will be patched regardless.

    Examples
    --------

    >>> from sklearnex import is_patched_instance
    >>> from sklearnex import patch_sklearn
    >>> from sklearn.linear_model import LinearRegression
    >>> is_patched_instance(LinearRegression())
    False
    >>> patch_sklearn()
    >>> from sklearn.linear_model import LinearRegression # now calls sklearnex
    >>> is_patched_instance(LinearRegression())
    True"""
    if preview:
        os.environ["SKLEARNEX_PREVIEW"] = "enabled_via_patch_sklearn"
    if not sklearn_check_version("1.0"):
        raise NotImplementedError(
            "Extension for Scikit-learn* patches apply "
            "for scikit-learn >= 1.0 only ..."
        )

    if global_patch:
        from sklearnex.glob.dispatcher import patch_sklearn_global

        patch_sklearn_global(name, verbose)

    from daal4py.sklearn import patch_sklearn as patch_sklearn_orig

    if _is_new_patching_available():
        for config in ["set_config", "get_config", "config_context"]:
            patch_sklearn_orig(
                config, verbose=False, deprecation=False, get_map=get_patch_map
            )
    if isinstance(name, list):
        for algorithm in name:
            patch_sklearn_orig(
                algorithm, verbose=False, deprecation=False, get_map=get_patch_map
            )
    else:
        patch_sklearn_orig(name, verbose=False, deprecation=False, get_map=get_patch_map)

    if verbose and sys.stderr is not None:
        sys.stderr.write(
            "Extension for Scikit-learn* enabled "
            "(https://github.com/uxlfoundation/scikit-learn-intelex)\n"
        )


def unpatch_sklearn(
    name: Optional[Union[str, list[str]]] = None, global_unpatch: bool = False
) -> None:
    """Unpatch scikit-learn.

    Unpatches the ``sklearn`` module, either as a whole or for selected estimators.

    .. Note
        If preview mode was enabled through ``patch_sklearn(preview=True)``, it will
        modify the environment variable ``SKLEARNEX_PREVIEW``, by deleting it.

    Parameters
    ----------
    name : str, list of str, or None
        Names of the desired estimators to check for patching status. Can
        pass a single instance name (e.g. ``"LogisticRegression"``), or a
        list of names (e.g. ``["LogisticRegression", "PCA"]``).

        If ``None``, will unpatch all the etimators that are patched.
    global_unpatch : bool
        Whether to unpatch the installed ``sklearn`` module itself, if patching had
        been applied to it (see :obj:`patch_sklearn`)."""
    if global_unpatch:
        from sklearnex.glob.dispatcher import unpatch_sklearn_global

        unpatch_sklearn_global()
    from daal4py.sklearn import unpatch_sklearn as unpatch_sklearn_orig

    if isinstance(name, list):
        for algorithm in name:
            unpatch_sklearn_orig(algorithm, get_map=get_patch_map)
    else:
        if _is_new_patching_available():
            for config in ["set_config", "get_config", "config_context"]:
                unpatch_sklearn_orig(config, get_map=get_patch_map)
        unpatch_sklearn_orig(name, get_map=get_patch_map)
    if os.environ.get("SKLEARNEX_PREVIEW") == "enabled_via_patch_sklearn":
        os.environ.pop("SKLEARNEX_PREVIEW")


def sklearn_is_patched(
    name: Optional[Union[str, list[str]]] = None, return_map: Optional[bool] = False
) -> Union[bool, dict[str, bool]]:
    """Check patching status.

    Checks whether patching of |sklearn| estimators has been applied, either as a whole
    or for a subset of estimators.

    Parameters
    ----------
    name : str, list of str, or None
        Names of the desired estimators to check for patching status. Can
        pass a single instance name (e.g. ``"LogisticRegression"``), or a
        list of names (e.g. ``["LogisticRegression", "PCA"]``).

        If ``None``, will check for patching status of all estimators.
    return_map : bool
        Whether to return per-estimator patching statuses, or just a single
        result, which will be ``True`` if all the estimators from ``name``
        are patched.

        .. important::

            The return map will contain names as used internally by the |sklearnex|.
            These names come in lower case, contain duplicates, and are a superset
            of the estimator names used by |sklearn| - for example, if applying full
            patching, will contain entries along the lines of ``"randomforestclassifier"``
            and ``"random_forest_classifier"``.

    Returns
    -------
    Check : bool or dict[str, bool]
        The patching status of the desired estimators, either as a whole, or
        on a per-estimator basis (output type controlled by ``return_map``)."""
    from daal4py.sklearn import sklearn_is_patched as sklearn_is_patched_orig

    if isinstance(name, list):
        if return_map:
            result: dict[str, bool] = {}
            for algorithm in name:
                try:
                    result[algorithm] = sklearn_is_patched_orig(
                        algorithm, get_map=get_patch_map
                    )
                except ValueError:
                    result[algorithm] = False
            return result
        else:
            is_patched = True
            for algorithm in name:
                try:
                    this_name_is_patched: bool = sklearn_is_patched_orig(
                        algorithm, get_map=get_patch_map
                    )
                except ValueError:
                    this_name_is_patched: bool = False
                is_patched = is_patched and this_name_is_patched
            return is_patched
    else:
        return sklearn_is_patched_orig(name, get_map=get_patch_map, return_map=return_map)


def is_patched_instance(instance: object) -> bool:
    """Check if given estimator instance is patched with scikit-learn-intelex.

    Parameters
    ----------
    instance : object
        Python object, usually a scikit-learn estimator instance.

    Returns
    -------
    Check : bool
        Boolean whether instance is a daal4py or sklearnex estimator.

    Examples
    --------

    >>> from sklearnex import is_patched_instance
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearnex.linear_model import LinearRegression as patched_LR
    >>> is_patched_instance(LinearRegression())
    False
    >>> is_patched_instance(patched_LR())
    True"""
    module = getattr(instance, "__module__", "")
    return ("daal4py" in module) or ("sklearnex" in module)

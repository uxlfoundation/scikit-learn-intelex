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
from daal4py.sklearn.monkeypatch.dispatcher import PatchMap


def _is_new_patching_available():
    return os.environ.get("OFF_ONEDAL_IFACE", "0") == "0" and daal_check_version(
        (2021, "P", 300)
    )


def _is_preview_enabled() -> bool:
    return "SKLEARNEX_PREVIEW" in os.environ


# Comment 2026-01-20: This file has been refactored from the original
# implementation. Initially, the patching map dicts from daal4py and
# sklearnex were somehow meant to share some keys and to be used in
# place of one another under some circumstances, and it appears at
# some point the sklearnex one was meant to inherit things from the
# daal4py one, but that is not the expected behavior anymore. Initially,
# the code tried to accomplish object sharing by merging LRU caches
# of the functions that produce the maps, and there might still be
# traces of this behavior in the refactored code, but by now it is
# expected that the patching map dict objects from both should be
# independent and the functions should be passed the right object
# when needed.


# Note: the keys of this dict are only used as internal IDs to keep
# track of things, and in the functions to check if a given function
# or class is patched. The keys can be arbitrary strings that do not
# necessarily correspond to module paths, but having the full paths
# and names of what they patch makes them easier to identify and debug.
@lru_cache(maxsize=None)
def get_patch_map_core(preview: bool = False) -> PatchMap:
    if preview:
        mapping = get_patch_map_core(preview=False)

        if _is_new_patching_available():
            import sklearn.covariance as covariance_module
            import sklearn.decomposition as decomposition_module
            from sklearn.covariance import (
                EmpiricalCovariance as EmpiricalCovariance_sklearn,
            )
            from sklearn.decomposition import IncrementalPCA as IncrementalPCA_sklearn

            # Preview classes for patching
            from .preview.covariance import (
                EmpiricalCovariance as EmpiricalCovariance_sklearnex,
            )
            from .preview.decomposition import IncrementalPCA as IncrementalPCA_sklearnex

            # Since the state of the lru_cache without preview cannot be
            # guaranteed to not have already enabled sklearnex algorithms
            # when preview is used, setting the mapping element[1] to None
            # should NOT be done. This may lose track of the unpatched
            # sklearn estimator or function.
            # Covariance
            preview_mapping = {
                "sklearn.covariance.EmpiricalCovariance": (
                    covariance_module,
                    "EmpiricalCovariance",
                    EmpiricalCovariance_sklearnex,
                    EmpiricalCovariance_sklearn,
                ),
                "sklearn.decomposition.IncrementalPCA": (
                    decomposition_module,
                    "IncrementalPCA",
                    IncrementalPCA_sklearnex,
                    IncrementalPCA_sklearn,
                ),
            }
            if daal_check_version((2024, "P", 1)):
                import sklearn.linear_model as linear_model_module
                from sklearn.linear_model import (
                    LogisticRegressionCV as LogisticRegressionCV_sklearn,
                )

                from .preview.linear_model import (
                    LogisticRegressionCV as LogisticRegressionCV_sklearnex,
                )

                preview_mapping["sklearn.linear_model.LogisticRegressionCV"] = (
                    (
                        linear_model_module,
                        "LogisticRegressionCV",
                        LogisticRegressionCV_sklearnex,
                        LogisticRegressionCV_sklearn,
                    ),
                )
            return mapping | preview_mapping

        return mapping

    # Comment 2026-01-20: This route is untested. It was meant to support
    # a situation in which the 'onedal' module is not compiled, and instead
    # the patching takes classes from daal4py, while still importing from
    # the sklearnex module. This is not tested in any kind of configurations.
    if not _is_new_patching_available():
        from daal4py.sklearn.monkeypatch.dispatcher import _get_map_of_algorithms

        return _get_map_of_algorithms()

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

    from sklearn.cluster import DBSCAN as DBSCAN_sklearn
    from sklearn.cluster import KMeans as KMeans_sklearn
    from sklearn.decomposition import PCA as PCA_sklearn
    from sklearn.ensemble import ExtraTreesClassifier as ExtraTreesClassifier_sklearn
    from sklearn.ensemble import ExtraTreesRegressor as ExtraTreesRegressor_sklearn
    from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier_sklearn
    from sklearn.ensemble import RandomForestRegressor as RandomForestRegressor_sklearn
    from sklearn.linear_model import ElasticNet as ElasticNet_sklearn
    from sklearn.linear_model import Lasso as Lasso_sklearn
    from sklearn.linear_model import LinearRegression as LinearRegression_sklearn
    from sklearn.linear_model import LogisticRegression as LogisticRegression_sklearn
    from sklearn.linear_model import Ridge as Ridge_sklearn
    from sklearn.manifold import TSNE as TSNE_sklearn
    from sklearn.neighbors import KNeighborsClassifier as KNeighborsClassifier_sklearn
    from sklearn.neighbors import KNeighborsRegressor as KNeighborsRegressor_sklearn
    from sklearn.neighbors import LocalOutlierFactor as LocalOutlierFactor_sklearn
    from sklearn.neighbors import NearestNeighbors as NearestNeighbors_sklearn
    from sklearn.svm import SVC as SVC_sklearn
    from sklearn.svm import SVR as SVR_sklearn
    from sklearn.svm import NuSVC as NuSVC_sklearn
    from sklearn.svm import NuSVR as NuSVR_sklearn

    if sklearn_check_version("1.4"):
        from sklearn.ensemble._gb import DummyRegressor as DummyRegressor_sklearn
    else:
        from sklearn.ensemble._gb_losses import DummyRegressor as DummyRegressor_sklearn
    from sklearn import config_context as config_context_sklearn
    from sklearn import get_config as get_config_sklearn
    from sklearn import set_config as set_config_sklearn
    from sklearn.metrics import pairwise_distances as pairwise_distances_sklearn
    from sklearn.metrics import roc_auc_score as roc_auc_score_sklearn
    from sklearn.model_selection import train_test_split as train_test_split_sklearn

    if sklearn_check_version("1.2.1"):
        from sklearn.utils.parallel import _FuncWrapper as _FuncWrapper_sklearn
        from sklearn.utils.parallel import get_config as parallel_get_config_sklearn
    else:
        from sklearn.utils.fixes import _FuncWrapper as _FuncWrapper_sklearn
        from sklearn.utils.fixes import get_config as parallel_get_config_sklearn

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

    mapping = {
        "sklearn.cluster.DBSCAN": (
            cluster_module,
            "DBSCAN",
            DBSCAN_sklearnex,
            DBSCAN_sklearn,
        ),
        "sklearn.cluster.KMeans": (
            cluster_module,
            "KMeans",
            KMeans_sklearnex,
            KMeans_sklearn,
        ),
        "sklearn.decomposition.PCA": (
            decomposition_module,
            "PCA",
            PCA_sklearnex,
            PCA_sklearn,
        ),
        "sklearn.svm.SVR": (svm_module, "SVR", SVR_sklearnex, SVR_sklearn),
        "sklearn.svm.SVC": (svm_module, "SVC", SVC_sklearnex, SVC_sklearn),
        "sklearn.svm.NuSVR": (svm_module, "NuSVR", NuSVR_sklearnex, NuSVR_sklearn),
        "sklearn.svm.NuSVC": (svm_module, "NuSVC", NuSVC_sklearnex, NuSVC_sklearn),
        "sklearn.linear_model.ElasticNet": (
            linear_model_module,
            "ElasticNet",
            ElasticNet_sklearnex,
            ElasticNet_sklearn,
        ),
        "sklearn.linear_model.Lasso": (
            linear_model_module,
            "Lasso",
            Lasso_sklearnex,
            Lasso_sklearn,
        ),
        "sklearn.linear_model.LinearRegression": (
            linear_model_module,
            "LinearRegression",
            LinearRegression_sklearnex,
            LinearRegression_sklearn,
        ),
        "sklearn.linear_model.LogisticRegression": (
            linear_model_module,
            "LogisticRegression",
            LogisticRegression_sklearnex,
            LogisticRegression_sklearn,
        ),
        "sklearn.linear_model.Ridge": (
            linear_model_module,
            "Ridge",
            Ridge_sklearnex,
            Ridge_sklearn,
        ),
        "sklearn.linear_model.IncrementalLinearRegression": (
            linear_model_module,
            "IncrementalLinearRegression",
            IncrementalLinearRegression_sklearnex,
            None,
        ),
        "sklearn.manifold.TSNE": (manifold_module, "TSNE", TSNE_sklearnex, TSNE_sklearn),
        "sklearn.metrics.pairwise_distances": (
            metrics_module,
            "pairwise_distances",
            pairwise_distances_sklearnex,
            pairwise_distances_sklearn,
        ),
        "sklearn.metrics.roc_auc_score": (
            metrics_module,
            "roc_auc_score",
            roc_auc_score_sklearnex,
            roc_auc_score_sklearn,
        ),
        "sklearn.model_selection.train_test_split": (
            model_selection_module,
            "train_test_split",
            train_test_split_sklearnex,
            train_test_split_sklearn,
        ),
        "sklearn.neighbors.KNeighborsClassifier": (
            neighbors_module,
            "KNeighborsClassifier",
            KNeighborsClassifier_sklearnex,
            KNeighborsClassifier_sklearn,
        ),
        "sklearn.neighbors.KNeighborsRegressor": (
            neighbors_module,
            "KNeighborsRegressor",
            KNeighborsRegressor_sklearnex,
            KNeighborsRegressor_sklearn,
        ),
        "sklearn.neighbors.NearestNeighbors": (
            neighbors_module,
            "NearestNeighbors",
            NearestNeighbors_sklearnex,
            NearestNeighbors_sklearn,
        ),
        "sklearn.neighbors.LocalOutlierFactor": (
            neighbors_module,
            "LocalOutlierFactor",
            LocalOutlierFactor_sklearnex,
            LocalOutlierFactor_sklearn,
        ),
        "sklearn.ensemble.ExtraTreesClassifier": (
            ensemble_module,
            "ExtraTreesClassifier",
            ExtraTreesClassifier_sklearnex,
            ExtraTreesClassifier_sklearn,
        ),
        "sklearn.ensemble.ExtraTreesRegressor": (
            ensemble_module,
            "ExtraTreesRegressor",
            ExtraTreesRegressor_sklearnex,
            ExtraTreesRegressor_sklearn,
        ),
        "sklearn.ensemble.RandomForestClassifier": (
            ensemble_module,
            "RandomForestClassifier",
            RandomForestClassifier_sklearnex,
            RandomForestClassifier_sklearn,
        ),
        "sklearn.ensemble.RandomForestRegressor": (
            ensemble_module,
            "RandomForestRegressor",
            RandomForestRegressor_sklearnex,
            RandomForestRegressor_sklearn,
        ),
        "sklearn.covariance.IncrementalEmpiricalCovariance": (
            covariance_module,
            "IncrementalEmpiricalCovariance",
            IncrementalEmpiricalCovariance_sklearnex,
            None,
        ),
        "sklearn.dummy.DummyRegressor": (
            dummy_module,
            "DummyRegressor",
            DummyRegressor_sklearnex,
            None,
        ),
        "sklearn.ensemble._gb_losses.DummyRegressor": (
            _gb_module,
            "DummyRegressor",
            DummyRegressor_sklearnex,
            DummyRegressor_sklearn,
        ),
        # These should be patched even if it applying to a single algorithm
        "sklearn.set_config": (
            base_module,
            "set_config",
            set_config_sklearnex,
            set_config_sklearn,
        ),
        "sklearn.get_config": (
            base_module,
            "get_config",
            get_config_sklearnex,
            get_config_sklearn,
        ),
        "sklearn.config_context": (
            base_module,
            "config_context",
            config_context_sklearnex,
            config_context_sklearn,
        ),
        # Comment 2026-01-20: The comment below was present in earlier code.
        # Whether it's true that is needed or not hasn't been verified.
        # --- end of comment 2026-01-20 ----
        # Necessary for proper work with multiple threads
        "sklearn.utils.parallel.get_config": (
            parallel_module,
            "get_config",
            get_config_sklearnex,
            parallel_get_config_sklearn,
        ),
        "sklearn.utils.parallel._funcwrapper": (
            parallel_module,
            "_FuncWrapper",
            _FuncWrapper_sklearnex,
            _FuncWrapper_sklearn,
        ),
    }

    if daal_check_version((2024, "P", 600)):
        mapping["sklearn.linear_model.IncrementalRidge"] = (
            linear_model_module,
            "IncrementalRidge",
            IncrementalRidge_sklearnex,
            None,
        )

    return mapping


# This is necessary to properly cache the patch_map when
# using preview.
def get_patch_map() -> PatchMap:
    preview: bool = _is_preview_enabled()
    return get_patch_map_core(preview=preview)


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
        ``"sklearn.linear_model.LogisticRegression"``), or a list of names (e.g. ``["sklearn.linear_model.LogisticRegression", "sklearn.decomposition.PCA"]``).

        If ``None``, will patch all the supported estimators.

        See the :doc:`algorithm support table <algorithms>` for more information.

        Note that functions related to :doc:`config contexts <config-contexts>` are
        always patched regardless of what's passed here.
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

    patch_map: PatchMap = get_patch_map()

    if name is not None and _is_new_patching_available():
        names_mandatory = [
            "sklearn.set_config",
            "sklearn.get_config",
            "sklearn.config_context",
            "sklearn.utils.parallel.get_config",
            "sklearn.utils.parallel._funcwrapper",
        ]
        for name_mandatory in names_mandatory:
            patch_sklearn_orig(
                name_mandatory, verbose=False, deprecation=False, map=patch_map
            )
    if isinstance(name, list):
        for algorithm in name:
            patch_sklearn_orig(algorithm, verbose=False, deprecation=False, map=patch_map)
    else:
        patch_sklearn_orig(name, verbose=False, deprecation=False, map=patch_map)

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
        pass a single instance name (e.g. ``"sklearn.linear_model.LogisticRegression"``), or a
        list of names (e.g. ``["sklearn.linear_model.LogisticRegression", "sklearn.decomposition.PCA"]``).

        If ``None``, will unpatch all the etimators that are patched.
    global_unpatch : bool
        Whether to unpatch the installed ``sklearn`` module itself, if patching had
        been applied to it (see :obj:`patch_sklearn`)."""
    if global_unpatch:
        from sklearnex.glob.dispatcher import unpatch_sklearn_global

        unpatch_sklearn_global()
    from daal4py.sklearn import unpatch_sklearn as unpatch_sklearn_orig

    patch_map: PatchMap = get_patch_map()

    if isinstance(name, list):
        for algorithm in name:
            unpatch_sklearn_orig(algorithm, map=patch_map)
    else:
        unpatch_sklearn_orig(name, map=patch_map)
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

    Returns
    -------
    Check : bool or dict[str, bool]
        The patching status of the desired estimators, either as a whole, or
        on a per-estimator basis (output type controlled by ``return_map``)."""
    from daal4py.sklearn import sklearn_is_patched as sklearn_is_patched_orig

    map = get_patch_map()

    if isinstance(name, list):
        if return_map:
            result: dict[str, bool] = {}
            for algorithm in name:
                result[algorithm] = sklearn_is_patched_orig(algorithm, map=map)
            return result
        else:
            is_patched = True
            for algorithm in name:
                is_patched = is_patched and sklearn_is_patched_orig(algorithm, map=map)
            return is_patched
    else:
        return sklearn_is_patched_orig(name, return_map=return_map, map=map)


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

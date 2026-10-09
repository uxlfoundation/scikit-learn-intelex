# ===============================================================================
# Copyright contributors to the oneDAL project
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

import numbers
import warnings

import numpy as np
from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN
from sklearn.utils._array_api import device as _device
from sklearn.utils.validation import check_array

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, is_sparse, sklearn_check_version
from onedal._device_offload import _transfer_to_host

if daal_check_version((2026, "P", 200)):
    from onedal.cluster import HDBSCAN as onedal_HDBSCAN
else:
    # the class below is still defined and still patched, so that this module
    # imports and the estimator stays visible to the tests that walk the
    # namespace. Nothing offloads: '_onedal_supported' reports every call
    # unsupported, which falls back to scikit-learn
    onedal_HDBSCAN = None

from .._device_offload import dispatch
from .._utils import PatchingConditionsChain
from ..base import oneDALEstimator
from ..utils._array_api import enable_array_api, get_namespace
from ..utils.validation import validate_data

# scikit-learn's own record layout for a dendrogram, the dtype of
# 'HDBSCAN._single_linkage_tree_'. Repeated rather than imported from the
# private 'sklearn.cluster._hdbscan._tree': a rename there would break the
# import of this module, and so all of the patching, where a change of the
# layout only breaks 'dbscan_clustering', loudly, on the typed memoryview
_SINGLE_LINKAGE_TREE_DTYPE = np.dtype(
    [
        ("left_node", np.int64),
        ("right_node", np.int64),
        ("value", np.float64),
        ("cluster_size", np.int64),
    ]
)


def _as_single_linkage_tree(tree):
    """Convert oneDAL's dendrogram table into scikit-learn's record array.

    Parameters
    ----------
    tree : ndarray of shape (n_samples - 1, 4)
        Merges in ascending distance order, as
        ``[left, right, distance, size]``.

    Returns
    -------
    tree : ndarray of shape (n_samples - 1,)
        The same merges, in '_SINGLE_LINKAGE_TREE_DTYPE'.
    """
    # a single observation has nothing to merge, and oneDAL reports no table
    # at all rather than an empty one, which arrives here as a flat array
    tree = np.reshape(tree, (-1, 4))
    result = np.empty(tree.shape[0], dtype=_SINGLE_LINKAGE_TREE_DTYPE)
    result["left_node"] = tree[:, 0]
    result["right_node"] = tree[:, 1]
    result["value"] = tree[:, 2]
    result["cluster_size"] = tree[:, 3]
    return result


# scikit-learn's names of the distances oneDAL computes, as oneDAL names them
_METRIC_ALIASES = {
    "euclidean": "euclidean",
    "l2": "euclidean",
    "manhattan": "manhattan",
    "l1": "manhattan",
    "cityblock": "manhattan",
    "chebyshev": "chebyshev",
    "infinity": "chebyshev",
    "minkowski": "minkowski",
    "p": "minkowski",
    "cosine": "cosine",
}

# the degrees for which the Minkowski distance is one of the dedicated metrics
_MINKOWSKI_DEGREES = {1: "manhattan", 2: "euclidean", np.inf: "chebyshev"}


def _onedal_metric(metric, metric_params, algorithm):
    """Map scikit-learn's distance onto the one oneDAL computes identically.

    Parameters
    ----------
    metric : str or callable
        The ``metric`` parameter of the estimator.
    metric_params : dict or None
        The ``metric_params`` parameter of the estimator.
    algorithm : str
        The ``algorithm`` parameter of the estimator.

    Returns
    -------
    metric : tuple of (str, float) or None
        oneDAL's metric and Minkowski degree, or None when scikit-learn's result
        or error cannot be reproduced.
    """
    if not isinstance(metric, str) or metric not in _METRIC_ALIASES:
        return None
    # the brute force passes the name to 'pairwise_distances', the trees to
    # 'DistanceMetric', and each of them takes names the other one rejects
    if algorithm == "brute" and metric in ("infinity", "p"):
        return None
    if algorithm in ("kd_tree", "ball_tree") and metric == "cosine":
        return None

    name = _METRIC_ALIASES[metric]
    metric_params = metric_params or {}
    if name != "minkowski":
        return None if metric_params else (name, 2.0)

    if not metric_params:
        # scipy, which serves the brute force, defaults to 'p=2' for 'minkowski',
        # the trees only do for 'p' and raise for 'minkowski'
        if algorithm != "brute" and metric == "minkowski":
            return None
        return "euclidean", 2.0
    if metric_params.keys() != {"p"}:
        return None
    p = metric_params["p"]
    if not isinstance(p, numbers.Real) or not p >= 1:
        return None
    if p in _MINKOWSKI_DEGREES:
        return _MINKOWSKI_DEGREES[p], 2.0
    return "minkowski", float(p)


def _non_finite_rows(X, xp):
    """Classify the samples the way scikit-learn's 'HDBSCAN.fit' does.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
        The data, in any namespace.
    xp : module
        The namespace of 'X'.

    Returns
    -------
    finite : ndarray of int
        Host indices of the samples with only finite values.
    infinite : ndarray of int
        Host indices of the samples with an infinite value and no NaN.
    missing : ndarray of int
        Host indices of the samples with a NaN, or with infinities of both signs.
    """
    # scikit-learn reduces its float64 copy of the data, the reduction is repeated
    # in float64 where that does not cost a conversion of the device data
    if isinstance(X, np.ndarray):
        reduced = np.sum(X, axis=1, dtype=np.float64)
    else:
        reduced = _transfer_to_host(xp.sum(X, axis=1))[1][0]
    reduced = np.asarray(reduced)
    return (
        np.isfinite(reduced).nonzero()[0],
        np.isinf(reduced).nonzero()[0],
        np.isnan(reduced).nonzero()[0],
    )


def _float64(xp, X):
    """scikit-learn's float64, or the dtype of 'X' on a device without it."""
    sycl_device = getattr(X, "sycl_device", None)
    if sycl_device is not None and not sycl_device.has_aspect_fp64:
        return X.dtype
    return xp.float64


@enable_array_api
@control_n_jobs(decorated_methods=["fit"])
class HDBSCAN(oneDALEstimator, _sklearn_HDBSCAN):
    __doc__ = _sklearn_HDBSCAN.__doc__

    # copied to keep 'control_n_jobs' from modifying scikit-learn's own constraints
    _parameter_constraints: dict = {**_sklearn_HDBSCAN._parameter_constraints}

    # scikit-learn's '__init__' is used as-is, all of its parameters are
    # forwarded to the onedal estimator or checked for oneDAL support

    _onedal_hdbscan = staticmethod(onedal_HDBSCAN)

    def _onedal_fit(self, X, queue=None):
        # oneDAL never writes into the data, so 'copy' has no effect here, but the
        # deprecation of its default has to be repeated for the offloaded path
        if (
            sklearn_check_version("1.8")
            and not sklearn_check_version("1.10")
            and self.copy == "warn"
        ):
            warnings.warn(
                "The default value of `copy` will change from False to True in 1.10."
                " Explicitly set a value for `copy` to silence this warning.",
                FutureWarning,
            )

        xp, _ = get_namespace(X)
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            dtype=[xp.float64, xp.float32],
            ensure_all_finite=False,
        )
        n_samples = X.shape[0]
        self._metric_params = self.metric_params or {}
        self._min_samples = (
            self.min_cluster_size if self.min_samples is None else self.min_samples
        )

        # as in scikit-learn, the samples with non-finite values are left out of
        # the clustering and labelled after it
        finite, infinite, missing = _non_finite_rows(X, xp)
        all_finite = len(finite) == n_samples
        if not all_finite:
            X = xp.take(X, xp.asarray(finite, device=_device(X)), axis=0)

        metric, degree = _onedal_metric(self.metric, self.metric_params, self.algorithm)
        if self.algorithm == "auto":
            # oneDAL computes the cosine distance only in its brute force method
            method = "brute_force" if metric == "cosine" else "kd_tree"
        elif self.algorithm == "brute":
            method = "brute_force"
        else:
            # 'kd_tree' and 'ball_tree' are named the same way in oneDAL
            method = self.algorithm

        onedal_params = {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self._min_samples,
            "metric": metric,
            "degree": degree,
            "alpha": self.alpha,
            "method": method,
            "leaf_size": self.leaf_size,
            "cluster_selection": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            # oneDAL takes zero as 'no limit on the size of a cluster'
            "max_cluster_size": self.max_cluster_size or 0,
            "store_centers": self.store_centers or "none",
        }
        self._onedal_estimator = self._onedal_hdbscan(**onedal_params)
        self._onedal_estimator.fit(X, queue=queue)

        # scikit-learn computes in float64 whatever the input is, and its outputs
        # are typed accordingly
        float64 = _float64(xp, X)
        labels = self._onedal_estimator.labels_
        probabilities = xp.astype(self._onedal_estimator.probabilities_, float64)

        # oneDAL reports no centers when it does not find any cluster, while
        # scikit-learn returns them empty. 'empty_like' allocates, so that the
        # estimator is not left holding a view on 'X'
        if self.store_centers in ("centroid", "both"):
            centroids = self._onedal_estimator.centroids_
            if centroids is None:
                centroids = xp.empty_like(X[:0, :])
            self.centroids_ = xp.astype(centroids, float64)
        if self.store_centers in ("medoid", "both"):
            medoids = self._onedal_estimator.medoids_
            if medoids is None:
                medoids = xp.empty_like(X[:0, :])
            self.medoids_ = xp.astype(medoids, float64)

        # the hierarchy the flat clustering above was cut out of, in the record
        # layout scikit-learn's own routines are typed on, so that a caller can
        # re-cut it at another distance -- see 'dbscan_clustering'
        tree = _as_single_linkage_tree(self._onedal_estimator.single_linkage_tree_)

        if all_finite:
            self._onedal_outliers = None
            self.labels_ = xp.astype(labels, xp.int64)
            self.probabilities_ = probabilities
            self._single_linkage_tree_ = tree
            return

        # imported here and not at module scope, see 'dbscan_clustering'
        from sklearn.cluster._hdbscan.hdbscan import (
            _OUTLIER_ENCODING,
            remap_single_linkage_tree,
        )

        self._single_linkage_tree_ = remap_single_linkage_tree(
            tree,
            {x: y for x, y in enumerate(finite)},
            non_finite=set(np.hstack([infinite, missing])),
        )
        # host indices of the outliers, which 'dbscan_clustering' labels again
        self._onedal_outliers = (infinite, missing)

        # the samples in the order finite, infinite, missing, and their places in
        # the input, so that the full outputs are one 'take' away: array API
        # namespaces do not have to support assignment through an index array
        order = np.concatenate([finite, infinite, missing])
        position = np.empty_like(order)
        position[order] = np.arange(n_samples)
        dev = _device(labels)
        position = xp.asarray(position, device=dev)

        def full(values, dtype, infinite_value, missing_value):
            parts = [
                xp.astype(values, dtype),
                xp.full(len(infinite), infinite_value, dtype=dtype, device=dev),
                xp.full(len(missing), missing_value, dtype=dtype, device=dev),
            ]
            return xp.take(xp.concat(parts), position, axis=0)

        # scikit-learn's own dtypes for this case, int32 for the labels
        self.labels_ = full(
            labels,
            xp.int32,
            _OUTLIER_ENCODING["infinite"]["label"],
            _OUTLIER_ENCODING["missing"]["label"],
        )
        self.probabilities_ = full(
            probabilities,
            float64,
            _OUTLIER_ENCODING["infinite"]["prob"],
            _OUTLIER_ENCODING["missing"]["prob"],
        )

    def _onedal_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.cluster.{class_name}.{method_name}"
        )
        if method_name != "fit":
            raise RuntimeError(
                f"Unknown method {method_name} in {self.__class__.__name__}"
            )

        # every case scikit-learn rejects falls back, for scikit-learn to raise
        # its own error
        X = data[0]
        dal_ready = patching_status.and_conditions(
            [
                (
                    onedal_HDBSCAN is not None,
                    "oneDAL version does not support HDBSCAN.",
                ),
                (
                    self.algorithm in ("auto", "brute", "kd_tree", "ball_tree"),
                    f"'{self.algorithm}' algorithm is not supported.",
                ),
                (
                    _onedal_metric(self.metric, self.metric_params, self.algorithm)
                    is not None,
                    f"'{self.metric}' metric with metric_params="
                    f"{self.metric_params} and '{self.algorithm}' algorithm is not "
                    "supported. Only the euclidean, manhattan, chebyshev, minkowski "
                    "(p >= 1) and cosine (brute force) distances are supported.",
                ),
                (not is_sparse(X), "X is sparse. Sparse input is not supported."),
            ]
        )
        if not dal_ready:
            return patching_status

        # the conversion only serves the checks below, the actual validation of
        # the data happens in '_onedal_fit'
        try:
            xp, _ = get_namespace(X)
            X = check_array(X, dtype="numeric", ensure_all_finite=False)
            finite, _, _ = _non_finite_rows(X, xp)
        except (TypeError, ValueError):
            patching_status.and_conditions(
                [(False, "X does not pass the validation of scikit-learn.")]
            )
            return patching_status

        n_finite = len(finite)
        min_samples = (
            self.min_cluster_size if self.min_samples is None else self.min_samples
        )
        patching_status.and_conditions(
            [
                (n_finite > 1, "X has fewer than two samples with finite values."),
                (
                    min_samples <= n_finite,
                    "min_samples is larger than the number of samples with finite "
                    "values in X.",
                ),
                (
                    # scikit-learn fails on it, indexing the finite samples with
                    # a mask over all of them
                    n_finite == X.shape[0] or self.store_centers is None,
                    "Centers of data with missing values or infinites are not "
                    "supported.",
                ),
            ]
        )
        return patching_status

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def fit(self, X, y=None):
        self._validate_params()
        # a refit that falls back must not leave 'dbscan_clustering' on the
        # outliers of an earlier offloaded fit
        for attribute in ("_onedal_estimator", "_onedal_outliers"):
            self.__dict__.pop(attribute, None)

        dispatch(
            self,
            "fit",
            {
                "onedal": self.__class__._onedal_fit,
                "sklearn": _sklearn_HDBSCAN.fit,
            },
            X,
        )

        return self

    fit.__doc__ = _sklearn_HDBSCAN.fit.__doc__

    def dbscan_clustering(self, cut_distance, min_cluster_size=5):
        if not hasattr(self, "_onedal_estimator"):
            return super().dbscan_clustering(cut_distance, min_cluster_size)

        # imported here and not at module scope on purpose: a rename in this
        # private module must not break the import of this one, and with it all
        # of the patching
        from sklearn.cluster._hdbscan._tree import labelling_at_cut

        labels = labelling_at_cut(
            self._single_linkage_tree_, cut_distance, min_cluster_size
        )
        # scikit-learn reads the outliers back from 'labels_', which may live on
        # a device, the ones of an offloaded fit are kept on the host
        if self._onedal_outliers is not None:
            from sklearn.cluster._hdbscan.hdbscan import _OUTLIER_ENCODING

            infinite, missing = self._onedal_outliers
            labels[infinite] = _OUTLIER_ENCODING["infinite"]["label"]
            labels[missing] = _OUTLIER_ENCODING["missing"]["label"]
        return labels

    dbscan_clustering.__doc__ = _sklearn_HDBSCAN.dbscan_clustering.__doc__

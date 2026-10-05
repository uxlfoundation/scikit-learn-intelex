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

import warnings

import numpy as np
from sklearn.cluster import HDBSCAN as _sklearn_HDBSCAN
from sklearn.utils.validation import _num_samples, check_array

from daal4py.sklearn._n_jobs_support import control_n_jobs
from daal4py.sklearn._utils import daal_check_version, is_sparse, sklearn_check_version

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
from ..utils.validation import assert_all_finite, validate_data

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
            ensure_all_finite=False,  # completed in offload check
        )

        if self.algorithm == "auto":
            # the kd-tree based neighbors search is the fastest option, but oneDAL
            # only implements it for a subset of the distances
            method = (
                "kd_tree"
                if self.metric in ("euclidean", "manhattan", "minkowski", "chebyshev")
                else "brute_force"
            )
        elif self.algorithm == "brute":
            method = "brute_force"
        else:
            # 'kd_tree' and 'ball_tree' are named the same way in oneDAL
            method = self.algorithm

        metric_params = self.metric_params or {}
        onedal_params = {
            # sklearn takes 'min_cluster_size' as 'min_samples' when unset
            "min_cluster_size": self.min_cluster_size,
            "min_samples": (
                self.min_cluster_size if self.min_samples is None else self.min_samples
            ),
            "metric": self.metric,
            # oneDAL validates the degree whatever the metric is, so
            # scikit-learn's 'p' is only taken where it has a meaning
            "degree": (
                metric_params.get("p", 2.0) if self.metric == "minkowski" else 2.0
            ),
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
        self.labels_ = self._onedal_estimator.labels_

        # oneDAL reports no centers when it does not find any cluster, while
        # scikit-learn returns them empty. 'empty_like' allocates, so that the
        # estimator is not left holding a view on 'X'
        if self.store_centers in ("centroid", "both"):
            centroids = self._onedal_estimator.centroids_
            self.centroids_ = xp.empty_like(X[:0, :]) if centroids is None else centroids
        if self.store_centers in ("medoid", "both"):
            medoids = self._onedal_estimator.medoids_
            self.medoids_ = xp.empty_like(X[:0, :]) if medoids is None else medoids

        self.probabilities_ = self._onedal_estimator.probabilities_

        # the hierarchy the flat clustering above was cut out of, in the record
        # layout scikit-learn's own routines are typed on, so that a caller can
        # re-cut it at another distance -- see 'dbscan_clustering'
        self._single_linkage_tree_ = _as_single_linkage_tree(
            self._onedal_estimator.single_linkage_tree_
        )

    def _onedal_supported(self, method_name, *data):
        class_name = self.__class__.__name__
        patching_status = PatchingConditionsChain(
            f"sklearn.cluster.{class_name}.{method_name}"
        )
        if method_name == "fit":
            X = data[0]
            # sklearn takes 'min_cluster_size' as 'min_samples' when unset
            min_samples = (
                self.min_cluster_size if self.min_samples is None else self.min_samples
            )
            dal_ready = patching_status.and_conditions(
                [
                    (
                        onedal_HDBSCAN is not None,
                        "oneDAL version does not support HDBSCAN.",
                    ),
                    (
                        # the metrics are named as scikit-learn names them, the
                        # mapping onto oneDAL's distances happens in '_onedal_fit'
                        self.metric
                        in (
                            "euclidean",
                            "manhattan",
                            "minkowski",
                            "chebyshev",
                            "cosine",
                        ),
                        f"'{self.metric}' metric is not supported. Only 'euclidean', "
                        "'manhattan', 'minkowski', 'chebyshev' and 'cosine' are "
                        "supported.",
                    ),
                    (
                        # oneDAL computes the cosine distance only in its brute
                        # force method
                        self.metric != "cosine" or self.algorithm in ("auto", "brute"),
                        "'cosine' metric is only supported by the 'auto' and 'brute' "
                        "algorithms.",
                    ),
                    (not is_sparse(X), "X is sparse. Sparse input is not supported."),
                    (
                        min_samples <= _num_samples(X),
                        "min_samples is larger than the number of samples in X.",
                    ),
                ]
            )
            if not dal_ready:
                return patching_status

            # sklearn labels non-finite samples as special outliers, while
            # oneDAL does not support them
            # the conversion is lax on purpose: it exists only to make the
            # finiteness check possible, the actual validation of the data
            # happens in '_onedal_fit'
            try:
                assert_all_finite(
                    check_array(
                        X,
                        dtype=None,
                        ensure_2d=False,
                        ensure_min_samples=0,
                        ensure_min_features=0,
                        accept_sparse=False,
                        ensure_all_finite=False,
                    )
                )
            except ValueError:
                patching_status.and_conditions(
                    [(False, "Missing values and infinites are not supported.")]
                )
            return patching_status
        raise RuntimeError(f"Unknown method {method_name} in {self.__class__.__name__}")

    _onedal_cpu_supported = _onedal_supported
    _onedal_gpu_supported = _onedal_supported

    def fit(self, X, y=None):
        self._validate_params()

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

        # scikit-learn's version then restores the labels of the samples it had
        # found to be infinite or missing during 'fit'. oneDAL does not take
        # such data in the first place, so there is nothing to restore, and
        # 'labels_', which may live on a device, does not have to be read back
        return labelling_at_cut(
            self._single_linkage_tree_, cut_distance, min_cluster_size
        )

    dbscan_clustering.__doc__ = _sklearn_HDBSCAN.dbscan_clustering.__doc__

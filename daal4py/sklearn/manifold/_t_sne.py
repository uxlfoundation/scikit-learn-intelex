# ==============================================================================
# Copyright 2020 Intel Corporation
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

# daal4py TSNE scikit-learn-compatible class

import warnings
from time import time

import numpy as np
from scipy.sparse import issparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE as BaseTSNE
from sklearn.manifold._t_sne import _joint_probabilities, _joint_probabilities_nn
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils import check_array, check_random_state
from sklearn.utils.validation import check_non_negative

import daal4py
from daal4py.sklearn._utils import (
    PatchingConditionsChain,
    daal_check_version,
    sklearn_check_version,
)

from .._n_jobs_support import control_n_jobs
from ..neighbors import NearestNeighbors
from ..utils.validation import validate_data


@control_n_jobs(decorated_methods=["fit"])
class TSNE(BaseTSNE):
    __doc__ = BaseTSNE.__doc__

    if sklearn_check_version("1.2"):
        _parameter_constraints: dict = {**BaseTSNE._parameter_constraints}

    def fit_transform(self, X, y=None):
        return super().fit_transform(X, y)

    def fit(self, X, y=None):
        return super().fit(X, y)

    def _daal_tsne(self, P, n_samples, X_embedded):
        """Runs t-SNE."""
        # t-SNE minimizes the Kullback-Leiber divergence of the Gaussians P
        # and the Student's t-distributions Q. The optimization algorithm that
        # we use is batch gradient descent with two stages:
        # * initial optimization with early exaggeration and momentum at 0.5
        # * final optimization with momentum at 0.8

        # N, nnz, n_iter_without_progress, n_iter
        size_iter = [
            [n_samples],
            [P.nnz],
            [self.n_iter_without_progress],
            [
                (
                    self.max_iter
                    if sklearn_check_version("1.7")
                    else (self._max_iter if sklearn_check_version("1.5") else self.n_iter)
                )
            ],
        ]

        # Pass params to daal4py backend
        if daal_check_version((2023, "P", 1)):
            size_iter.extend(
                [
                    [
                        (
                            self._EXPLORATION_MAX_ITER
                            if sklearn_check_version("1.5")
                            else self._EXPLORATION_N_ITER
                        )
                    ],
                    [self._N_ITER_CHECK],
                ]
            )

        size_iter = np.array(size_iter, dtype=P.dtype)

        params = np.array(
            [
                [self.early_exaggeration],
                [self._learning_rate],
                [self.min_grad_norm],
                [self.angle],
            ],
            dtype=P.dtype,
        )
        results = np.zeros((3, 1), dtype=P.dtype)  # curIter, error, gradNorm

        if P.dtype == np.float64:
            daal4py.daal_tsne_gradient_descent(
                X_embedded, P, size_iter, params, results, 0
            )
        elif P.dtype == np.float32:
            daal4py.daal_tsne_gradient_descent(
                X_embedded, P, size_iter, params, results, 1
            )
        else:
            raise ValueError("unsupported dtype of 'P' matrix")

        # Save the final number of iterations
        self.n_iter_ = int(results[0][0])

        # Save Kullback-Leiber divergence
        self.kl_divergence_ = results[1][0]

        return X_embedded

    def _fit(self, X, skip_num_points=0):
        """Private function to fit the model using X as training data."""
        if isinstance(self.init, str) and self.init == "warn":
            warnings.warn(
                "The default initialization in TSNE will change "
                "from 'random' to 'pca' in 1.2.",
                FutureWarning,
            )
            self._init = "random"
        else:
            self._init = self.init

        if isinstance(self._init, str) and self._init == "pca" and issparse(X):
            raise TypeError(
                "PCA initialization is currently not supported "
                "with the sparse input matrix. Use "
                'init="random" instead.'
            )

        if self.method not in ["barnes_hut", "exact"]:
            raise ValueError("'method' must be 'barnes_hut' or 'exact'")
        if self.angle < 0.0 or self.angle > 1.0:
            raise ValueError("'angle' must be between 0.0 - 1.0")
        if self.learning_rate == "warn":
            warnings.warn(
                "The default learning rate in TSNE will change "
                "from 200.0 to 'auto' in 1.2.",
                FutureWarning,
            )
            self._learning_rate = 200.0
        else:
            self._learning_rate = self.learning_rate
        if self._learning_rate == "auto":
            self._learning_rate = X.shape[0] / self.early_exaggeration / 4
            self._learning_rate = np.maximum(self._learning_rate, 50)
        else:
            if not (self._learning_rate > 0):
                raise ValueError(
                    "'learning_rate' must be a positive number " "or 'auto'."
                )
        # rename attribute for compatibility with sklearn>=1.2
        if sklearn_check_version("1.2"):
            self.learning_rate_ = self._learning_rate

        if hasattr(self, "square_distances"):
            if sklearn_check_version("1.1"):
                if self.square_distances != "deprecated":
                    warnings.warn(
                        "The parameter `square_distances` has not effect "
                        "and will be removed in version 1.3.",
                        FutureWarning,
                    )
            else:
                if self.square_distances not in [True, "legacy"]:
                    raise ValueError("'square_distances' must be True or 'legacy'.")
                if self.metric != "euclidean" and self.square_distances is not True:
                    warnings.warn(
                        "'square_distances' has been introduced in 0.24 to help phase "
                        "out legacy squaring behavior. The 'legacy' setting will be "
                        "removed in 1.1 (renaming of 0.26), and the default setting "
                        "will be changed to True. In 1.3, 'square_distances' will be "
                        "removed altogether, and distances will be squared by "
                        "default. Set 'square_distances'=True to silence this "
                        "warning.",
                        FutureWarning,
                    )

        if self.method == "barnes_hut":
            X = validate_data(
                self,
                X,
                accept_sparse=["csr"],
                ensure_min_samples=2,
                dtype=[np.float32, np.float64],
            )
        else:
            X = validate_data(
                self,
                X,
                accept_sparse=["csr", "csc", "coo"],
                dtype=[np.float32, np.float64],
            )

        if self.metric == "precomputed":
            if isinstance(self._init, str) and self._init == "pca":
                raise ValueError(
                    'The parameter init="pca" cannot be '
                    'used with metric="precomputed".'
                )
            if X.shape[0] != X.shape[1]:
                raise ValueError("X should be a square distance matrix")

            check_non_negative(
                X,
                "TSNE.fit(). With metric='precomputed', X "
                "should contain positive distances.",
            )

            if self.method == "exact" and issparse(X):
                raise TypeError(
                    'TSNE with method="exact" does not accept sparse '
                    'precomputed distance matrix. Use method="barnes_hut" '
                    "or provide the dense distance matrix."
                )

        if self.method == "barnes_hut" and self.n_components > 3:
            raise ValueError(
                "'n_components' should be inferior to 4 for the "
                "barnes_hut algorithm as it relies on "
                "quad-tree or oct-tree."
            )
        random_state = check_random_state(self.random_state)

        if self.early_exaggeration < 1.0:
            raise ValueError(
                "early_exaggeration must be at least 1, but is {}".format(
                    self.early_exaggeration
                )
            )

        if not sklearn_check_version("1.2"):
            if self.n_iter < 250:
                raise ValueError("n_iter should be at least 250")

        n_samples = X.shape[0]

        neighbors_nn = None
        if self.method == "exact":
            # Retrieve the distance matrix, either using the precomputed one or
            # computing it.
            if self.metric == "precomputed":
                distances = X
            else:
                if self.verbose:
                    print("[t-SNE] Computing pairwise distances...")

                if self.metric == "euclidean":
                    # Euclidean is squared here, rather than using **= 2,
                    # because euclidean_distances already calculates
                    # squared distances, and returns np.sqrt(dist) for
                    # squared=False.
                    # Also, Euclidean is slower for n_jobs>1, so don't set here
                    distances = pairwise_distances(X, metric=self.metric, squared=True)
                else:
                    metric_params_ = {}
                    if sklearn_check_version("1.1"):
                        metric_params_ = self.metric_params or {}
                    distances = pairwise_distances(
                        X, metric=self.metric, n_jobs=self.n_jobs, **metric_params_
                    )

            if np.any(distances < 0):
                raise ValueError(
                    "All distances should be positive, the " "metric given is not correct"
                )

            if (
                self.metric != "euclidean"
                and getattr(self, "square_distances", True) is True
            ):
                distances **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities(distances, self.perplexity, self.verbose)
            assert np.all(np.isfinite(P)), "All probabilities should be finite"
            assert np.all(P >= 0), "All probabilities should be non-negative"
            assert np.all(P <= 1), (
                "All probabilities should be less " "or then equal to one"
            )

        else:
            # Compute the number of nearest neighbors to find.
            # LvdM uses 3 * perplexity as the number of neighbors.
            # In the event that we have very small # of points
            # set the neighbors to n - 1.
            n_neighbors = min(n_samples - 1, int(3.0 * self.perplexity + 1))

            if self.verbose:
                print("[t-SNE] Computing {} nearest neighbors...".format(n_neighbors))

            # Find the nearest neighbors for every point
            knn = None
            if sklearn_check_version("1.1"):
                knn = NearestNeighbors(
                    algorithm="auto",
                    n_jobs=self.n_jobs,
                    n_neighbors=n_neighbors,
                    metric=self.metric,
                    metric_params=self.metric_params,
                )
            else:
                knn = NearestNeighbors(
                    algorithm="auto",
                    n_jobs=self.n_jobs,
                    n_neighbors=n_neighbors,
                    metric=self.metric,
                )
            t0 = time()
            knn.fit(X)
            duration = time() - t0
            if self.verbose:
                print(
                    "[t-SNE] Indexed {} samples in {:.3f}s...".format(n_samples, duration)
                )

            t0 = time()
            distances_nn = knn.kneighbors_graph(mode="distance")
            duration = time() - t0
            if self.verbose:
                print(
                    "[t-SNE] Computed neighbors for {} samples "
                    "in {:.3f}s...".format(n_samples, duration)
                )

            # Free the memory used by the ball_tree
            del knn

            if (
                getattr(self, "square_distances", True) is True
                or self.metric == "euclidean"
            ):
                # knn return the euclidean distance but we need it squared
                # to be consistent with the 'exact' method. Note that the
                # the method was derived using the euclidean method as in the
                # input space. Not sure of the implication of using a different
                # metric.
                distances_nn.data **= 2

            # compute the joint probability distribution for the input space
            P = _joint_probabilities_nn(distances_nn, self.perplexity, self.verbose)

        if isinstance(self._init, np.ndarray):
            X_embedded = self._init
        elif self._init == "pca":
            pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=random_state,
            )
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            warnings.warn(
                "The PCA initialization in TSNE will change to "
                "have the standard deviation of PC1 equal to 1e-4 "
                "in 1.2. This will ensure better convergence.",
                FutureWarning,
            )
        elif self._init == "random":
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.randn(n_samples, self.n_components).astype(
                np.float32
            )
        else:
            raise ValueError("'init' must be 'pca', 'random', or " "a numpy array")

        # Degrees of freedom of the Student's t-distribution. The suggestion
        # degrees_of_freedom = n_components - 1 comes from
        # "Learning a Parametric Embedding by Preserving Local Structure"
        # Laurens van der Maaten, 2009.
        degrees_of_freedom = max(self.n_components - 1, 1)

        _patching_status = PatchingConditionsChain("sklearn.manifold.TSNE._tsne")
        _patching_status.and_conditions(
            [
                (
                    self.method == "barnes_hut",
                    'Used t-SNE method is not "barnes_hut" which is the only supported.',
                ),
                (self.n_components == 2, "Number of components != 2."),
                (self.verbose == 0, "Verbose mode is set."),
                (
                    daal_check_version((2021, "P", 600)),
                    "oneDAL version is lower than 2021.6.",
                ),
            ]
        )
        _dal_ready = _patching_status.get_status(logs=True)

        if _dal_ready:
            X_embedded = check_array(X_embedded, dtype=[np.float32, np.float64])
            return self._daal_tsne(P, n_samples, X_embedded=X_embedded)
        return self._tsne(
            P,
            degrees_of_freedom,
            n_samples,
            X_embedded=X_embedded,
            neighbors=neighbors_nn,
            skip_num_points=skip_num_points,
        )

    fit.__doc__ = BaseTSNE.fit.__doc__
    fit_transform.__doc__ = BaseTSNE.fit_transform.__doc__

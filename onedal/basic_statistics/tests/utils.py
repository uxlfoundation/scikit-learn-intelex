# ===============================================================================
# Copyright 2024 Intel Corporation
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

import numpy as np


# Compute unbiased variation for the columns of array-like X
def variation(X):
    X_mean = np.mean(X, axis=0)
    if np.all(X_mean):
        # Avoid division by zero
        return np.std(X, axis=0, ddof=1) / X_mean
    else:
        return np.array(
            [
                x / y if y != 0 else np.nan
                for x, y in zip(np.std(X, axis=0, ddof=1), X_mean)
            ]
        )


options_and_tests = {
    "sum": (lambda X: np.sum(X, axis=0), (5e-4, 1e-7)),
    "min": (lambda X: np.min(X, axis=0), (1e-7, 1e-7)),
    "max": (lambda X: np.max(X, axis=0), (1e-7, 1e-7)),
    "mean": (lambda X: np.mean(X, axis=0), (5e-7, 1e-7)),
    # sklearnex computes unbiased variance and standard deviation that is why ddof=1
    "variance": (lambda X: np.var(X, axis=0, ddof=1), (2e-4, 1e-7)),
    "variation": (lambda X: variation(X), (1e-3, 1e-6)),
    "sum_squares": (lambda X: np.sum(np.square(X), axis=0), (2e-4, 1e-7)),
    "sum_squares_centered": (
        lambda X: np.sum(np.square(X - np.mean(X, axis=0)), axis=0),
        (1e-3, 1e-7),
    ),
    "standard_deviation": (lambda X: np.std(X, axis=0, ddof=1), (2e-3, 1e-7)),
    "second_order_raw_moment": (lambda X: np.mean(np.square(X), axis=0), (1e-6, 1e-7)),
}

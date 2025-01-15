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

options_and_tests = {
    "sum": (lambda X: np.sum(X, axis=0), (5e-4, 1e-7)),
    "min": (lambda X: np.min(X, axis=0), (1e-7, 1e-7)),
    "max": (lambda X: np.max(X, axis=0), (1e-7, 1e-7)),
    "mean": (lambda X: np.mean(X, axis=0), (5e-7, 1e-7)),
    "variance": (lambda X: np.var(X, axis=0), (2e-3, 2e-3)),
    "variation": (lambda X: np.std(X, axis=0) / np.mean(X, axis=0), (5e-2, 5e-2)),
    "sum_squares": (lambda X: np.sum(np.square(X), axis=0), (2e-4, 1e-7)),
    "sum_squares_centered": (
        lambda X: np.sum(np.square(X - np.mean(X, axis=0)), axis=0),
        (2e-4, 1e-7),
    ),
    "standard_deviation": (lambda X: np.std(X, axis=0), (2e-3, 2e-3)),
    "second_order_raw_moment": (lambda X: np.mean(np.square(X), axis=0), (1e-6, 1e-7)),
}

options_and_tests_sparse = {
    "sum": (lambda X: np.sum(X, axis=0), (5e-4, 1e-7)),
    "min": (lambda X: np.min(X, axis=0), (1e-7, 1e-7)),
    #    TODO: There is a bug in oneDAL's max computations on GPU
    #    "max": (lambda X: np.max(X, axis=0), (1e-7, 1e-7)),
    "mean": (lambda X: np.mean(X, axis=0), (5e-7, 1e-7)),
    "variance": (lambda X: np.var(X, axis=0), (2e-3, 2e-3)),
    "variation": (lambda X: np.std(X, axis=0) / np.mean(X, axis=0), (0.1, 0.1)),
    #    TODO: There is a bug in oneDAL's max computations on GPU
    #    "sum_squares": (lambda X: np.sum(np.square(X), axis=0), (2e-4, 1e-7)),
    "sum_squares_centered": (
        lambda X: np.sum(np.square(X - np.mean(X, axis=0)), axis=0),
        (0.1, 1e-6),
    ),
    "standard_deviation": (lambda X: np.std(X, axis=0), (1e-2, 2e-3)),
    "second_order_raw_moment": (lambda X: np.mean(np.square(X), axis=0), (1e-6, 1e-7)),
}

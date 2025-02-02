# ==============================================================================
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
# ==============================================================================

import numpy as np
import numpy.random as rand
import pytest
import scipy.sparse as sp

from onedal.tests.utils._dataframes_support import (
    _convert_to_dataframe,
    get_dataframes_and_queues,
)
from onedal.utils.validation import assert_all_finite


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape",
    [
        [16, 2048],
        [65539],  # 2**16 + 3,
        [1000, 1000],
        [
            3,
        ],
    ],
)
@pytest.mark.parametrize("allow_nan", [False, True])
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues("numpy,dpnp,dpctl,array_api")
)
def test_sum_infinite_actually_finite(dtype, shape, allow_nan, dataframe, queue):
    X = np.empty(shape, dtype=dtype)
    X.fill(np.finfo(dtype).max)
    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)
    assert_all_finite(X, allow_nan=allow_nan)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "shape",
    [
        [16, 2048],
        [65539],  # 2**16 + 3,
        [1000, 1000],
        [
            3,
        ],
    ],
)
@pytest.mark.parametrize("allow_nan", [False, True])
@pytest.mark.parametrize("check", ["inf", "NaN", None])
@pytest.mark.parametrize("seed", [0, 123456])
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues("numpy,dpnp,dpctl,array_api")
)
def test_assert_finite_random_location(
    dtype, shape, allow_nan, check, seed, dataframe, queue
):
    rand.seed(seed)
    X = rand.uniform(high=np.finfo(dtype).max, size=shape).astype(dtype)

    if check:
        loc = rand.randint(0, X.size - 1)
        X.reshape((-1,))[loc] = float(check)

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    if check is None or (allow_nan and check == "NaN"):
        assert_all_finite(X, allow_nan=allow_nan)
    else:
        msg_err = "Input contains " + ("infinity" if allow_nan else "NaN, infinity") + "."
        with pytest.raises(ValueError, match=msg_err):
            assert_all_finite(X, allow_nan=allow_nan)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("allow_nan", [False, True])
@pytest.mark.parametrize("check", ["inf", "NaN", None])
@pytest.mark.parametrize("seed", [0, 123456])
@pytest.mark.parametrize(
    "dataframe, queue", get_dataframes_and_queues("numpy,dpnp,dpctl,array_api")
)
def test_assert_finite_random_shape_and_location(
    dtype, allow_nan, check, seed, dataframe, queue
):
    lb, ub = 2, 1048576  # ub is 2^20
    rand.seed(seed)
    X = rand.uniform(high=np.finfo(dtype).max, size=rand.randint(lb, ub)).astype(dtype)

    if check:
        loc = rand.randint(0, X.size - 1)
        X[loc] = float(check)

    X = _convert_to_dataframe(X, sycl_queue=queue, target_df=dataframe)

    if check is None or (allow_nan and check == "NaN"):
        assert_all_finite(X, allow_nan=allow_nan)
    else:
        msg_err = "Input contains " + ("infinity" if allow_nan else "NaN, infinity") + "."
        with pytest.raises(ValueError, match=msg_err):
            assert_all_finite(X, allow_nan=allow_nan)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("allow_nan", [False, True])
@pytest.mark.parametrize("check", ["inf", "NaN", None])
@pytest.mark.parametrize("seed", [0, 123456])
def test_assert_finite_sparse(dtype, allow_nan, check, seed):
    lb, ub = 2, 2056
    rand.seed(seed)
    X = sp.random(
        rand.randint(lb, ub),
        rand.randint(lb, ub),
        format="csr",
        dtype=dtype,
        random_state=rand.default_rng(seed),
    )

    if check:
        locx = rand.randint(0, X.data.shape[0] - 1)
        X.data[locx] = float(check)

    if check is None or (allow_nan and check == "NaN"):
        assert_all_finite(X, allow_nan=allow_nan)
    else:
        msg_err = "Input contains " + ("infinity" if allow_nan else "NaN, infinity") + "."
        with pytest.raises(ValueError, match=msg_err):
            assert_all_finite(X, allow_nan=allow_nan)

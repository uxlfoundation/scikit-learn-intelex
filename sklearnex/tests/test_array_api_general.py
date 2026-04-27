# ==============================================================================
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
# ==============================================================================

import array_api_strict
import numpy as np

from sklearnex import config_context
from sklearnex.linear_model import LinearRegression


def test_non_writeable_arrays():
    rng = np.random.default_rng(seed=123)
    X = rng.random(size=(20, 4))
    y = rng.random(size=X.shape[0])
    X.flags.writeable = False
    y.flags.writeable = False
    Xs = array_api_strict.asarray(X)
    ys = array_api_strict.asarray(y)
    with config_context(array_api_dispatch=True):
        model = LinearRegression().fit(Xs, ys)
        _ = model.predict(Xs)
        _ = model.score(Xs, ys)

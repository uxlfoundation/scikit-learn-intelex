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

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.cluster.tests.common import generate_clustered_data

from onedal.cluster import Louvain

# Common networking dataset https://en.wikipedia.org/wiki/Zachary%27s_karate_club
_karate_club = sp.csr_array(
    (
        np.ones((156,), dtype=np.float64),
        np.array(
            [
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                10,
                11,
                12,
                13,
                17,
                19,
                21,
                31,
                0,
                2,
                3,
                7,
                13,
                17,
                19,
                21,
                30,
                0,
                1,
                3,
                7,
                8,
                9,
                13,
                27,
                28,
                32,
                0,
                1,
                2,
                7,
                12,
                13,
                0,
                6,
                10,
                0,
                6,
                10,
                16,
                0,
                4,
                5,
                16,
                0,
                1,
                2,
                3,
                0,
                2,
                30,
                32,
                33,
                2,
                33,
                0,
                4,
                5,
                0,
                0,
                3,
                0,
                1,
                2,
                3,
                33,
                32,
                33,
                32,
                33,
                5,
                6,
                0,
                1,
                32,
                33,
                0,
                1,
                33,
                32,
                33,
                0,
                1,
                32,
                33,
                25,
                27,
                29,
                32,
                33,
                25,
                27,
                31,
                23,
                24,
                31,
                29,
                33,
                2,
                23,
                24,
                33,
                2,
                31,
                33,
                23,
                26,
                32,
                33,
                1,
                8,
                32,
                33,
                0,
                24,
                25,
                28,
                32,
                33,
                2,
                8,
                14,
                15,
                18,
                20,
                22,
                23,
                29,
                30,
                31,
                33,
                8,
                9,
                13,
                14,
                15,
                18,
                19,
                20,
                22,
                23,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
            ]
        ),
        np.array(
            [
                0,
                16,
                25,
                35,
                41,
                44,
                48,
                52,
                56,
                61,
                63,
                66,
                67,
                69,
                74,
                76,
                78,
                80,
                82,
                84,
                87,
                89,
                91,
                93,
                98,
                101,
                104,
                106,
                110,
                113,
                117,
                121,
                127,
                139,
                156,
            ]
        ),
    )
)

def test_Louvain_karate_club():
    est = Louvain()
    est.fit(_karate_club)

# ==============================================================================
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
# ==============================================================================

from importlib.machinery import SourceFileLoader
from pathlib import Path

utils_path = Path(__file__).parent.parent / "utils"

readcsv = SourceFileLoader("readcsv", str(utils_path / "readcsv.py")).load_module()

np_read_csv = readcsv.np_read_csv
pd_read_csv = readcsv.pd_read_csv
csr_read_csv = readcsv.csr_read_csv
read_next = readcsv.read_next

__all__ = [
    "np_read_csv",
    "pd_read_csv",
    "csr_read_csv",
    "read_next",
]

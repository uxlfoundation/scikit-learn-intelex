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

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

utils_path = Path(__file__).parent.parent / "utils"

spec = spec_from_file_location("readcsv", str(utils_path / "readcsv.py"))
readcsv = module_from_spec(spec)
spec.loader.exec_module(readcsv)

np_read_csv = readcsv.np_read_csv
pd_read_csv = readcsv.pd_read_csv
csr_read_csv = readcsv.csr_read_csv

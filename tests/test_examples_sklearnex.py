#===============================================================================
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
#===============================================================================

import os
import subprocess
import sys
import unittest
from daal4py.sklearn._utils import get_daal_version
test_path = os.path.abspath(os.path.dirname(__file__))
unittest_data_path = os.path.join(test_path, "unittest_data")
examples_path = os.path.join(
    os.path.dirname(test_path), "examples", "sklearnex")
sys.path.insert(0, examples_path)
os.chdir(examples_path)

python_executable = subprocess.run(
    ['/usr/bin/which', 'python'], check=True,
    capture_output=True).stdout.decode().strip()

# First item is major version - 2021,
# second is minor+patch - 0110,
# third item is status - B
sklearnex_version = get_daal_version()
print('oneDAL version:', sklearnex_version)


class TestsklearnexExamples(unittest.TestCase):
    '''Class for testing sklernex examples'''
    def test_examples_in_directory(self):
        # Get a list of all Python files in the examples directory
        files = [f for f in os.listdir(examples_path) if f.endswith(".py")]

        # Iterate over each file and run it as a test case
        for file in files:
            with self.subTest(file=file):
                # Run the script and capture its exit code
                process = subprocess.run(
                    [python_executable, os.path.join(examples_path, file)],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    check=True)
                exit_code = process.returncode

                # Assert that the exit code is 0
                self.assertEqual(exit_code, 0)


if __name__ == '__main__':
    unittest.main()

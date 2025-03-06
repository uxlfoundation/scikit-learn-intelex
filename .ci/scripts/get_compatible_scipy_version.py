#! /usr/bin/env python
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
from sys import argv, exit, stderr
from sys import version_info as python_version

if len(argv) == 1:
    from daal4py.sklearn._utils import sklearn_check_version
else:
    # This is necessary to prevent import errors from sklearnex and by extension
    # sklearn caused by initial scipy and/or numpy versions that are installed.
    # It is written in a way to be `packaging` independent. This branch occurs
    # when a sklearn version is given to the script externally.
    def sklearn_check_version(ver):
        ver = [int(i) if i != "" else 0 for i in ver.split(".")[:2]]
        sk_ver = [int(i) if i != "" else 0 for i in str(argv[1]).split(".")[:2]]
        return sk_ver[0] > ver[0] or (sk_ver[0] == ver[0] and sk_ver[1] >= ver[1])


if sklearn_check_version("1.4"):
    print("Scipy version is not specified for this sklearn/python version.", file=stderr)
    print("scipy")
elif sklearn_check_version("1.3") or python_version[1] > 11:
    if python_version[1] > 8:
        print("scipy==1.12.*")
    else:
        print("scipy==1.11.*")
elif sklearn_check_version("1.2") or python_version[1] > 10:
    if python_version[1] > 9:
        print("scipy==1.12.*")
    else:
        print("scipy==1.9.*")
elif sklearn_check_version("1.1"):
    print("scipy==1.8.*")
elif sklearn_check_version("1.0"):
    print("scipy==1.7.*")
else:
    print(
        "Scipy version defaults to not specified "
        "for this outdated sklearn/python version.",
        file=stderr,
    )
    print("scipy")

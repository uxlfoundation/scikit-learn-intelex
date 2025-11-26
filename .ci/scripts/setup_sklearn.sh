#!/bin/bash
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

# Args:
# 1 - sklearn version (optional, default=main)

repo_dir=$( dirname $( dirname $( dirname "${BASH_SOURCE[0]}" ) ) )
cd $repo_dir

sklearn_version=${1:-main}

if [ "$sklearn_version" == "main" ]; then
    # remove sklearn version from test requirements file
    sed -i.bak -E "s/scikit-learn==[0-9a-zA-Z.]*/scikit-learn/" requirements-test.txt
    # install sklearn build dependencies
    pip install threadpoolctl joblib scipy
    # install sklearn from main branch of git repo
    pip install git+https://github.com/scikit-learn/scikit-learn.git@main
else
    sed -i.bak -E "s/scikit-learn==[0-9a-zA-Z.]*/scikit-learn==${sklearn_version}.*/" requirements-test.txt
fi

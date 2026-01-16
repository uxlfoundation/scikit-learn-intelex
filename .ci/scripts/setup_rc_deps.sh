#!/bin/bash
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

# This script is meant to install release candidate versions of
# dependencies, if any have release candidates with big changes
# that need to be tried at a given moment. The list is meant to
# be dynamic and change as needed over time.

repo_dir=$( dirname $( dirname $( dirname "${BASH_SOURCE[0]}" ) ) )
cd $repo_dir

if [[ "${RC_DEPS}" == "1" ]]; then
    pip install --upgrade --pre pandas==3.*
fi

#!/bin/bash
#===============================================================================
# Copyright 2021 Intel Corporation
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

SAMPLES_DIR=sources/samples

# remove the samples folder if it exists
if [ -d "$SAMPLES_DIR" ]; then rm -Rf $SAMPLES_DIR; fi

# create a samples folder
mkdir $SAMPLES_DIR

# copy jupyter notebooks
cd ..
rsync -a --exclude='daal4py_data_science.ipynb' examples/notebooks/*.ipynb doc/$SAMPLES_DIR

# build the documentation
cd doc
export SPHINXOPTS="-W" # used by sphinx-build
export O=${SPHINXOPTS} # makefile overrides SPHINXOPTS

# Build comes in two variants:
# - As a standalone doc (for local development and CI)
# - As a versioned build (for the deployed docs), triggerable by
#   passing argument '--gh-pages'.
# In the first case, it will generate a page under '_build' with
# an 'html' folder and a separate 'doctrees'. Only the 'html' part
# is needed to render the docs locally.
# In the second case, it will generate a versioned entry (year.month)
# under '_build', with the doctrees inside that versioned folder.
# Those are directly copyable to the 'gh-pages' branch to be deployed.
if [[ "$*" == *"--gh-pages"* ]]; then
    export DOC_VERSION=$(python -c "from sources.conf import version; print(version)")
    export SPHINXPROJ=scikit-learn-intelex
    export BUILDDIR=_build
    export SOURCEDIR=sources

    sphinx-build -b html $SPHINXOPTS $SOURCEDIR $BUILDDIR/$SPHINXPROJ/$DOC_VERSION
    echo "<meta http-equiv=\"refresh\" content=\"0; URL='/$SPHINXPROJ/$DOC_VERSION/'\" / >" >> $BUILDDIR/$SPHINXPROJ/index.html
else
    make html
fi

#Run the link-checker after build avoid rate limit errors
sphinx-build -b linkcheck -j auto $SPHINXOPTS sources _build/linkcheck
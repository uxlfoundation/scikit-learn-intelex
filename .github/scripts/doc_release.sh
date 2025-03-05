#!/bin/bash

#===============================================================================
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
#===============================================================================

# Ensure the build directory exists
BUILD_DIR="doc/_build/scikit-learn-intelex"
if [ ! -d "$BUILD_DIR" ]; then
    echo "::error: Documentation build directory not found!"
    exit 1
fi

# Copy built documentation to a temp location
DEPLOY_DIR="/tmp/gh-pages-deploy"
mkdir -p "$DEPLOY_DIR"
cp -R "$BUILD_DIR"/* "$DEPLOY_DIR"
ls $DEPLOY_DIR

# Checkout gh-pages branch
if ! git checkout gh-pages; then
    echo "::error:: Could not checkout gh-pages branch!"
    exit 1
fi

# Move the new versioned folder to the correct location
rm -Rf latest
cp -R "$DEPLOY_DIR/$SHORT_DOC_VERSION" "$SHORT_DOC_VERSION"
cp -R "$DEPLOY_DIR/$SHORT_DOC_VERSION" latest
cp "$DEPLOY_DIR/index.html" .
if ! diff -r "$SHORT_DOC_VERSION" latest > /dev/null; then
    echo "::error: Content mismatch between $SHORT_DOC_VERSION and latest directories"
    echo "Differences found:"
    diff -r "$SHORT_DOC_VERSION" latest
    exit 1
fi

# Generate versions.json by scanning for year.month folders
rm -f doc/versions.json
mkdir -p doc
echo "[" > doc/versions.json
# Add latest entry first
echo '  {"name": "latest", "version": "'$SHORT_DOC_VERSION'", "url": "/scikit-learn-intelex/latest/"},' >> doc/versions.json
# Add all year.month folders
for version in $(ls -d [0-9][0-9][0-9][0-9].[0-9]* 2>/dev/null || true); do
  echo '  {"name": "'$version'", "version": "'$version'", "url": "/scikit-learn-intelex/'$version'/"},'
done | sort -rV >> doc/versions.json
# Remove trailing comma and close array
sed -i '$ s/,$//' doc/versions.json
echo "]" >> doc/versions.json
cat doc/versions.json

# Commit and push changes
git add -A "$SHORT_DOC_VERSION"
git add -A latest
git add doc/versions.json
git add index.html
git commit . -m "Automatic doc update for version $DOC_VERSION"
git push origin gh-pages

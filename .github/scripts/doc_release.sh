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

rm -rf _site
mkdir -p _site
STORAGE_BRANCH="doc_archive"

##### Get archived docs from tar.gz artifacts #####
if git ls-remote --heads origin $STORAGE_BRANCH | grep -q $STORAGE_BRANCH; then
    echo "$STORAGE_BRANCH branch exists, checking for archived artifacts..."
    git fetch origin $STORAGE_BRANCH:$STORAGE_BRANCH
    git worktree add archive_sync $STORAGE_BRANCH

    # Get most recent tar.gz in archive branch
    LATEST_ARTIFACT=$(ls -t archive_sync/*.tar.gz 2>/dev/null | head -n 1)
    if [ -n "$LATEST_ARTIFACT" ] && [ -f "$LATEST_ARTIFACT" ]; then
        echo "Extracting most recent archived documentation from $LATEST_ARTIFACT..."
        mkdir -p temp_extract
        tar -xzf "$LATEST_ARTIFACT" -C temp_extract
        rsync -av --ignore-existing temp_extract/ _site/
        rm -rf temp_extract
    else
        echo "No tar.gz artifacts found in $STORAGE_BRANCH branch."
    fi
    git worktree remove archive_sync --force
else
    echo "$STORAGE_BRANCH branch does not exist, skipping archive sync."
fi

##### Get potential new docs from gh-pages #####
if git ls-remote --heads origin gh-pages | grep -q gh-pages; then
    echo "gh-pages branch exists, setting up worktree for sync..."
    git fetch origin gh-pages:gh-pages
    git worktree add gh-pages gh-pages
    rsync -av --ignore-existing gh-pages/ _site/
    git worktree remove gh-pages --force
else
    echo "gh-pages branch does not exist, skipping sync."
fi

##### Prepare new doc #####
# Copy the new built version to _site
mkdir -p _site/$SHORT_DOC_VERSION
cp -R doc/_build/scikit-learn-intelex/$SHORT_DOC_VERSION/* _site/$SHORT_DOC_VERSION/

# Update latest
rm -rf _site/latest
mkdir -p _site/latest
cp -R doc/_build/scikit-learn-intelex/$SHORT_DOC_VERSION/* _site/latest/

# Copy index.html
cp doc/_build/scikit-learn-intelex/index.html _site/

# Generate versions.json
mkdir -p _site/doc
echo "[" > _site/versions.json
# Add latest entry first
echo '  {"name": "latest", "version": "'$SHORT_DOC_VERSION'", "url": "/scikit-learn-intelex/latest/"},' >> _site/versions.json
# Add all year.month folders
for version in $(ls -d _site/[0-9][0-9][0-9][0-9].[0-9]* 2>/dev/null || true); do
    version=$(basename "$version")
    echo '  {"name": "'$version'", "version": "'$version'", "url": "/scikit-learn-intelex/'$version'/"},'
done | sort -rV >> _site/versions.json
# Remove trailing comma and close array
sed -i '$ s/,$//' _site/versions.json
echo "]" >> _site/versions.json

# Display the content for verification
ls -la _site/
cat _site/versions.json
git checkout -- .github/scripts/doc_release.sh

##### Archive Current state to a tar.gz #####
echo "Archiving version $SHORT_DOC_VERSION to branch $STORAGE_BRANCH..."
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Create a tar.gz artifact of the entire _site directory
ARTIFACT_NAME="doc-site-${SHORT_DOC_VERSION}.tar.gz"
echo "Creating documentation artifact: $ARTIFACT_NAME"
tar -czf "$ARTIFACT_NAME" -C _site .

# Check if storage branch exists
if git ls-remote --heads origin "$STORAGE_BRANCH" | grep -q "$STORAGE_BRANCH"; then
    echo "Storage branch exists, fetching it..."
    git fetch origin $STORAGE_BRANCH
    git checkout $STORAGE_BRANCH
else
    echo "Creating new storage branch..."
    # Create an empty orphan branch
    git checkout --orphan $STORAGE_BRANCH
    git rm -rf .
    git commit --allow-empty -m "Initialize doc archive branch"
    git push origin $STORAGE_BRANCH
fi

# Move the artifact to the storage branch
mv "$ARTIFACT_NAME" .

# Commit & push
git add "$ARTIFACT_NAME"
git commit -m "Archive complete documentation site for version $SHORT_DOC_VERSION"
git push origin $STORAGE_BRANCH

# Return to original branch
git checkout $CURRENT_BRANCH
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

BUILD_DIR="doc/_build/scikit-learn-intelex"
STORAGE_BRANCH="doc_archive"

# Check if TEMP_DOC_FOLDER is set
if [ -z "$TEMP_DOC_FOLDER" ]; then
    echo "::error::TEMP_DOC_FOLDER environment variable is not set!"
    exit 1
fi

# Ensure the build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "::error: Documentation build directory not found!"
    exit 1
fi

rm -rf $TEMP_DOC_FOLDER
mkdir -p $TEMP_DOC_FOLDER

##### Get archived version folders from doc_archive and gh-pages #####
# Function to sync content from a branch to the temp folder
sync_from_branch() {
    local branch_name=$1
    
    if git ls-remote --heads origin $branch_name | grep -q $branch_name; then
        echo "$branch_name branch exists, syncing content..."
        git fetch origin $branch_name:$branch_name
        git worktree add branch_sync $branch_name
        rsync -av --ignore-existing branch_sync/ $TEMP_DOC_FOLDER/
        git worktree remove branch_sync --force
    else
        echo "$branch_name branch does not exist, skipping sync."
    fi
}
sync_from_branch $STORAGE_BRANCH
sync_from_branch "gh-pages"

##### Prepare new doc #####
# Copy the new built version to $TEMP_DOC_FOLDER
mkdir -p $TEMP_DOC_FOLDER/$SHORT_DOC_VERSION
cp -R doc/_build/scikit-learn-intelex/$SHORT_DOC_VERSION/* $TEMP_DOC_FOLDER/$SHORT_DOC_VERSION/

# Update latest
rm -rf $TEMP_DOC_FOLDER/latest
mkdir -p $TEMP_DOC_FOLDER/latest
cp -R doc/_build/scikit-learn-intelex/$SHORT_DOC_VERSION/* $TEMP_DOC_FOLDER/latest/

# Copy index.html
cp doc/_build/scikit-learn-intelex/index.html $TEMP_DOC_FOLDER/

# Generate versions.json
mkdir -p $TEMP_DOC_FOLDER
echo "[" > $TEMP_DOC_FOLDER/versions.json
# Add latest entry first
echo '  {"name": "latest", "version": "'$SHORT_DOC_VERSION'", "url": "/scikit-learn-intelex/latest/"},' >> $TEMP_DOC_FOLDER/versions.json
# Add all year.month folders
for version in $(ls -d $TEMP_DOC_FOLDER/[0-9][0-9][0-9][0-9].[0-9]* 2>/dev/null || true); do
    version=$(basename "$version")
    echo '  {"name": "'$version'", "version": "'$version'", "url": "/scikit-learn-intelex/'$version'/"},'
done | sort -rV >> $TEMP_DOC_FOLDER/versions.json
# Remove trailing comma and close array
sed -i '$ s/,$//' $TEMP_DOC_FOLDER/versions.json
echo "]" >> $TEMP_DOC_FOLDER/versions.json

# Display the content for verification
ls -la $TEMP_DOC_FOLDER/
cat $TEMP_DOC_FOLDER/versions.json
git checkout -- .github/scripts/doc_release.sh

##### Archive to doc_archive branch #####
echo "Archiving version $SHORT_DOC_VERSION to branch $STORAGE_BRANCH..."
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Check if storage branch exists
if git ls-remote --heads origin "$STORAGE_BRANCH" | grep -q "$STORAGE_BRANCH"; then
    echo "Storage branch exists, fetching it..."
    git fetch origin $STORAGE_BRANCH
    git checkout $STORAGE_BRANCH
    
    # Add only the new version directory
    mkdir -p $SHORT_DOC_VERSION
    rsync -av $TEMP_DOC_FOLDER/$SHORT_DOC_VERSION/ $SHORT_DOC_VERSION/    
    git add $SHORT_DOC_VERSION
    git commit -m "Add documentation for version $SHORT_DOC_VERSION"
else
    echo "Creating new storage branch with all current versions..."
    # Create an empty orphan branch
    git checkout --orphan $STORAGE_BRANCH
    git rm -rf .

    # Copy only version folders
    for version_dir in $(find $TEMP_DOC_FOLDER -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9].[0-9]*" 2>/dev/null); do
        version=$(basename "$version_dir")
        mkdir -p $version
        rsync -av "$version_dir/" $version/
    done
    
    # Git only add verison folders
    git add -- [0-9][0-9][0-9][0-9].[0-9]* 
    git commit -m "Initialize doc archive branch with all versions"
fi

# Push changes
git push origin $STORAGE_BRANCH

# Return to original branch
git checkout $CURRENT_BRANCH
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

# Parse command line arguments
IS_DEV_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            IS_DEV_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if TEMP_DOC_FOLDER is set
if [ -z "$TEMP_DOC_FOLDER" ]; then
    echo "::error::TEMP_DOC_FOLDER environment variable is not set!"
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
if [ "$IS_DEV_MODE" = true ]; then
    # Dev mode: Build and update dev documentation
    echo "Building dev documentation..."
    
    # Ensure the build directory exists
    if [ ! -d "$BUILD_DIR" ]; then
        echo "::error: Documentation build directory not found!"
        exit 1
    fi
    
    # Copy dev documentation from build directory
    if [ ! -d "$BUILD_DIR/dev" ]; then
        echo "::error: Dev documentation not found in build directory!"
        exit 1
    fi
    
    rm -rf $TEMP_DOC_FOLDER/dev
    mkdir -p $TEMP_DOC_FOLDER/dev
    cp -R $BUILD_DIR/dev/* $TEMP_DOC_FOLDER/dev/
else
    # Release mode: Copy from dev to create new release version
    echo "Creating release documentation for version $SHORT_DOC_VERSION from dev..."
    
    if [ ! -d "$TEMP_DOC_FOLDER/dev" ]; then
        echo "::error: Dev documentation not found! Cannot create release."
        exit 1
    fi
    
    # Create versioned folder from dev
    mkdir -p $TEMP_DOC_FOLDER/$SHORT_DOC_VERSION
    cp -R $TEMP_DOC_FOLDER/dev/* $TEMP_DOC_FOLDER/$SHORT_DOC_VERSION/

    # Update latest
    rm -rf $TEMP_DOC_FOLDER/latest
    mkdir -p $TEMP_DOC_FOLDER/latest
    cp -R $TEMP_DOC_FOLDER/dev/* $TEMP_DOC_FOLDER/latest/
fi

# Copy root index.html if it exists (only in dev mode, as release mode doesn't build)
if [ "$IS_DEV_MODE" = true ] && [ -f "$BUILD_DIR/index.html" ]; then
    cp $BUILD_DIR/index.html $TEMP_DOC_FOLDER/
fi

# Generate versions.json
echo "[" > $TEMP_DOC_FOLDER/versions.json
# Add dev entry if it exists
if [ -d "$TEMP_DOC_FOLDER/dev" ]; then
    echo '  {"name": "dev (next release)", "version": "dev", "url": "/scikit-learn-intelex/dev/"},' >> $TEMP_DOC_FOLDER/versions.json
fi
# Add latest entry if it exists
if [ -d "$TEMP_DOC_FOLDER/latest" ]; then
    LATEST_VERSION=$(find $TEMP_DOC_FOLDER -mindepth 1 -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9].[0-9]*" | sort -rV | head -n 1 | xargs basename 2>/dev/null || echo "latest")
    echo '  {"name": "latest", "version": "'$LATEST_VERSION'", "url": "/scikit-learn-intelex/latest/"},' >> $TEMP_DOC_FOLDER/versions.json
fi
# Add all versioned folders
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
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

if [ "$IS_DEV_MODE" = true ]; then
    # Dev mode: Archive dev documentation
    echo "Archiving dev documentation to branch $STORAGE_BRANCH..."
    
    if git ls-remote --heads origin "$STORAGE_BRANCH" | grep -q "$STORAGE_BRANCH"; then
        echo "Storage branch exists, updating dev documentation..."
        git fetch origin $STORAGE_BRANCH
        git checkout $STORAGE_BRANCH
        
        rm -rf dev
        mkdir -p dev
        rsync -av $TEMP_DOC_FOLDER/dev/ dev/
        git add dev
        git commit -m "Update dev documentation" || echo "No changes to commit for dev"
    else
        echo "Creating new storage branch with dev documentation..."
        git checkout --orphan $STORAGE_BRANCH
        git rm -rf .
        
        mkdir -p dev
        rsync -av $TEMP_DOC_FOLDER/dev/ dev/
        git add dev
        git commit -m "Initialize doc archive branch with dev documentation"
    fi
    
    git push origin $STORAGE_BRANCH
    git checkout $CURRENT_BRANCH
else
    # Release mode: Archive versioned documentation
    echo "Archiving version $SHORT_DOC_VERSION to branch $STORAGE_BRANCH..."

    if git ls-remote --heads origin "$STORAGE_BRANCH" | grep -q "$STORAGE_BRANCH"; then
        echo "Storage branch exists, fetching it..."
        git fetch origin $STORAGE_BRANCH
        git checkout $STORAGE_BRANCH
        
        mkdir -p $SHORT_DOC_VERSION
        rsync -av $TEMP_DOC_FOLDER/$SHORT_DOC_VERSION/ $SHORT_DOC_VERSION/    
        git add $SHORT_DOC_VERSION
        git commit -m "Add documentation for version $SHORT_DOC_VERSION"
    else
        echo "Creating new storage branch..."
        git checkout --orphan $STORAGE_BRANCH
        git rm -rf .

        for version_dir in $(find $TEMP_DOC_FOLDER -maxdepth 1 -type d -name "[0-9][0-9][0-9][0-9].[0-9]*" 2>/dev/null); do
            version=$(basename "$version_dir")
            mkdir -p $version
            rsync -av "$version_dir/" $version/
        done
        
        git add -- [0-9][0-9][0-9][0-9].[0-9]* 
        git commit -m "Initialize doc archive branch with all versions"
    fi

    git push origin $STORAGE_BRANCH
    git checkout $CURRENT_BRANCH
fi
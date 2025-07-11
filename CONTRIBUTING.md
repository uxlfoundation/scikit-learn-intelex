<!--
******************************************************************************
* Copyright 2022 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/-->

# How to Contribute

As an open source project, we welcome community contributions to Extension for Scikit-learn. 
This document explains how to participate in project conversations, log bugs and enhancement requests, and submit code patches.

## Licensing 

Extension for Scikit-learn uses the [Apache 2.0 License](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/LICENSE). 
By contributing to the project, you agree to the license and copyright terms and release your own contributions under these terms. 

## Pull Requests 

No anonymous contributions are accepted. The name in the commit message Signed-off-by line and your email must match the change authorship information. 

Make sure your ``.gitconfig`` is set up correctly so you can use `git commit -s` for signing your patches: 

`git config --global user.name "Kate Developer"`

`git config --global user.email kate.developer@company.com`
 
### Before Contributing Changes

* Make sure you can build the product and run all the tests with your patch. 
* For a larger feature, provide a relevant test. 
* Document your code. Our project uses reStructuredText for documentation.  
* For new file(s), specify the appropriate copyright year in the first line. 
* Submit a pull request into the main branch. 

Continuous Integration (CI) testing is enabled for the repository. Your pull request must pass all checks before it can be merged. We will review your contribution and may provide feedback to guide you if any additional fixes or modifications are necessary. When reviewed and accepted, your pull request will be merged into our GitHub repository. 

## Code Style

We use [black](https://black.readthedocs.io/en/stable/) version 24.1.1 and [isort](https://pycqa.github.io/isort/) version 5.13.2 formatters for Python* code. The line length is 90 characters; use default options otherwise. You can find the linter configuration in [.pyproject.toml](https://github.com/uxlfoundation/scikit-learn-intelex/blob/main/pyproject.toml).

A GitHub* Action verifies if your changes comply with the output of the auto-formatting tools.

Optionally, you can install pre-commit hooks that do the formatting for you. For this, run from the top level of the repository:

## Ideas

If you want to contribute but do not know where to start we maintain a [public list](https://uxlfoundation.github.io/scikit-learn-intelex/latest/ideas.html) of projects which include difficulty and effort in our documentation.  These ideas have linked issues on GitHub where you can message us for next steps.

```bash
pip install pre-commit
pre-commit install
```

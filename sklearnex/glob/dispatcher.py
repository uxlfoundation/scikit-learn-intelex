# ===============================================================================
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
# ===============================================================================


def get_patch_str(name=None, verbose=True):
    return f"""try:
    # TEMP. FIX: sklearnex.patch_sklearn imports sklearn beforehand
    # when it didn't initialized _threadpool_controller required for
    # pairwise distances dispatching during imports.
    # Manually setting and deleting _threadpool_controller during patch fixes it.
    import sklearn
    from threadpoolctl import ThreadpoolController
    sklearn._threadpool_controller = ThreadpoolController()
    from sklearnex import patch_sklearn
    patch_sklearn(name={str(name)}, verbose={str(verbose)})
    del patch_sklearn, sklearn._threadpool_controller
except ImportError:
    pass"""


def get_patch_str_re():
    return r"""\ntry:
    \# TEMP. FIX: sklearnex.patch_sklearn imports sklearn beforehand
    \# when it didn't initialized _threadpool_controller required for
    \# pairwise distances dispatching during imports.
    \# Manually setting and deleting _threadpool_controller during patch fixes it.
    import sklearn
    from threadpoolctl import ThreadpoolController
    sklearn._threadpool_controller = ThreadpoolController\(\)
    from sklearnex import patch_sklearn
    patch_sklearn\(name=.*, verbose=.*\)
    del patch_sklearn, sklearn._threadpool_controller
except ImportError:
    pass\n"""


def patch_sklearn_global(name=None, verbose=True):
    import os
    import re

    try:
        import sklearn
    except ImportError:
        raise ImportError("Scikit-learn could not be imported. Nothing to patch\n")

    init_file_path = sklearn.__file__
    distributor_file_path = os.path.join(
        os.path.dirname(init_file_path), "_distributor_init.py"
    )

    with open(distributor_file_path, "r", encoding="utf-8") as distributor_file:
        lines = distributor_file.read()
        if re.search(get_patch_str_re(), lines):
            lines = re.sub(get_patch_str_re(), "", lines)

    with open(distributor_file_path, "w", encoding="utf-8") as distributor_file:
        distributor_file.write(lines + "\n" + get_patch_str(name, verbose) + "\n")
        print(
            "Scikit-learn was successfully globally patched"
            " by Extension for Scikit-learn"
        )
        return


def unpatch_sklearn_global():
    import os
    import re

    try:
        import sklearn
    except ImportError:
        raise ImportError("Scikit-learn could not be imported. Nothing to unpatch\n")

    init_file_path = sklearn.__file__
    distributor_file_path = os.path.join(
        os.path.dirname(init_file_path), "_distributor_init.py"
    )

    with open(distributor_file_path, "r", encoding="utf-8") as distributor_file:
        lines = distributor_file.read()
        if not re.search(get_patch_str_re(), lines):
            print("Nothing to unpatch: Scikit-learn is not patched\n")
            return
        lines = re.sub(get_patch_str_re(), "", lines)

    with open(distributor_file_path, "w", encoding="utf-8") as distributor_file:
        distributor_file.write(lines)
        print("Scikit-learn was successfully globally unpatched")

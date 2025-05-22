# ==============================================================================
# Copyright 2014 Intel Corporation
# Copyright 2024 Fujitsu Limited
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

import platform

if "Windows" in platform.system():
    import os
    import site
    import sys

    arch_dir = platform.machine()
    plt_dict = {"x86_64": "intel64", "AMD64": "intel64", "aarch64": "arm"}
    arch_dir = plt_dict[arch_dir] if arch_dir in plt_dict else arch_dir

    current_path = os.path.dirname(__file__)
    path_to_env = site.getsitepackages()[0]
    path_to_libs = os.path.join(path_to_env, "Library", "bin")
    if sys.version_info.minor >= 8:
        if "DALROOT" in os.environ:
            dal_root = os.environ["DALROOT"]
            dal_root_redist = os.path.join(dal_root, "redist", arch_dir)
            if os.path.exists(dal_root_redist):
                os.add_dll_directory(dal_root_redist)
                os.environ["PATH"] = dal_root_redist + os.pathsep + os.environ["PATH"]
        if "TBBROOT" in os.environ:
            tbb_root = os.environ["TBBROOT"]
            tbb_root_redist = os.path.join(tbb_root, "bin")
            if os.path.exists(tbb_root_redist):
                os.add_dll_directory(tbb_root_redist)
                os.environ["PATH"] = tbb_root_redist + os.pathsep + os.environ["PATH"]

        try:
            os.add_dll_directory(path_to_libs)
        except FileNotFoundError:
            pass

    os.environ["PATH"] = path_to_libs + os.pathsep + os.environ["PATH"]

try:
    from daal4py._daal4py import *
    from daal4py._daal4py import (
        __has_dist__,
        _get__daal_link_version__,
        _get__daal_run_version__,
        _get__version__,
    )
except ImportError as e:
    s = str(e)
    if "libfabric" in s:
        raise ImportError(
            s + "\n\nActivating your conda environment or sourcing mpivars."
            "[c]sh/psxevars.[c]sh may solve the issue.\n"
        )

    raise

from . import mb, sklearn

__all__ = ["mb", "sklearn"]

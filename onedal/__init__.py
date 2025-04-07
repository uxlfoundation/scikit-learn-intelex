# ==============================================================================
# Copyright 2021 Intel Corporation
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

from daal4py.sklearn._utils import daal_check_version


class Backend:
    """Encapsulates the oneDAL pybind11 modules and provides a unified interface to it together with additional properties about dpc/spmd policies"""

    def __init__(self, backend_module, is_dpc, is_spmd):
        self.backend = backend_module
        self.is_dpc = is_dpc
        self.is_spmd = is_spmd

    # accessing the instance will return the backend_module
    def __getattr__(self, name):
        return getattr(self.backend, name)

    def __repr__(self) -> str:
        return f"Backend({self.backend}, is_dpc={self.is_dpc}, is_spmd={self.is_spmd})"


if "Windows" in platform.system():
    import os
    import site
    import sys

    arch_dir = platform.machine()
    plt_dict = {"x86_64": "intel64", "AMD64": "intel64", "aarch64": "arm"}
    arch_dir = plt_dict[arch_dir] if arch_dir in plt_dict else arch_dir
    path_to_env = site.getsitepackages()[0]
    path_to_libs = os.path.join(path_to_env, "Library", "bin")
    if sys.version_info.minor >= 8:
        if "DALROOT" in os.environ:
            dal_root_redist = os.path.join(os.environ["DALROOT"], "redist", arch_dir)
            if os.path.exists(dal_root_redist):
                os.add_dll_directory(dal_root_redist)
        try:
            os.add_dll_directory(path_to_libs)
        except FileNotFoundError:
            pass
    os.environ["PATH"] = path_to_libs + os.pathsep + os.environ["PATH"]


try:
    # use dpc backend if available
    import onedal._onedal_py_dpc

    _dpc_backend = Backend(onedal._onedal_py_dpc, is_dpc=True, is_spmd=False)

    _host_backend = None
except ImportError:
    # fall back to host backend
    _dpc_backend = None

    import onedal._onedal_py_host

    _host_backend = Backend(onedal._onedal_py_host, is_dpc=False, is_spmd=False)

try:
    # also load spmd backend if available
    import onedal._onedal_py_spmd_dpc

    _spmd_backend = Backend(onedal._onedal_py_spmd_dpc, is_dpc=True, is_spmd=True)
except ImportError:
    _spmd_backend = None

# if/elif/else layout required for pylint to realize _default_backend cannot be None
if _dpc_backend is not None:
    _default_backend = _dpc_backend
elif _host_backend is not None:
    _default_backend = _host_backend
else:
    raise ImportError("No oneDAL backend available")

# Core modules to export
__all__ = [
    "_host_backend",
    "_default_backend",
    "_dpc_backend",
    "_spmd_backend",
    "covariance",
    "decomposition",
    "ensemble",
    "neighbors",
    "primitives",
    "svm",
]

# Additional features based on version checks
if daal_check_version((2023, "P", 100)):
    __all__ += ["basic_statistics", "linear_model"]
if daal_check_version((2023, "P", 200)):
    __all__ += ["cluster"]

# Exports if SPMD backend is available
if _spmd_backend is not None:
    __all__ += ["spmd"]
    if daal_check_version((2023, "P", 100)):
        __all__ += [
            "spmd.basic_statistics",
            "spmd.decomposition",
            "spmd.linear_model",
            "spmd.neighbors",
        ]
    if daal_check_version((2023, "P", 200)):
        __all__ += ["spmd.cluster"]

__version__ = "2199.9.9"

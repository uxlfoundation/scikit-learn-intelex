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

import pathlib as _pathlib
import platform

from daal4py.sklearn._utils import daal_check_version


class Backend:

    def __init__(self, backend_module, is_dpc, is_spmd):
        """A unified interface to an available oneDAL pybind11 module.

        This class encapsulates a oneDAL pybind11 module allowing for
        dynamic access of module objects. This simplifies method and
        attribute access in sklearnex without aliasing in sys.modules.
        It contains additional attributes for inspection of the pybind11
        module type (i.e. dpc or spmd) for use in policy creation.

        Parameters
        ----------
            backend_module : oneDAL pybind11 module
                Pybind11 module to be encapsulated.

            is_dpc : bool
                Flag describing if the module is Data Parallel C++-enabled.

            is_spmd : bool
                Flag describing if the module is single program, multiple
                data enabled.

        Returns
        -------
            self : Backend
                Encapsulated oneDAL pybind11 module.
        """

        self.backend = backend_module
        self.is_dpc = is_dpc
        self.is_spmd = is_spmd

    # accessing the instance will return the backend_module
    def __getattr__(self, name):
        return getattr(self.backend, name)

    def __repr__(self) -> str:
        return f"Backend({self.backend}, is_dpc={self.is_dpc}, is_spmd={self.is_spmd})"


def _backend_binary_present(prefix: str) -> bool:
    """Return True if a backend extension binary with the given prefix exists."""
    return any(_pathlib.Path(__file__).parent.glob(f"{prefix}*"))


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

        if _backend_binary_present("_onedal_py_dpc"):
            for dep_root in ["CMPLR_ROOT", "MKLROOT"]:
                if dep_root in os.environ:
                    dep_root_dir = os.path.join(os.environ[dep_root], "bin")
                    if os.path.exists(dep_root_dir):
                        os.add_dll_directory(dep_root_dir)

        try:
            os.add_dll_directory(path_to_libs)
        except FileNotFoundError:
            pass
    os.environ["PATH"] = path_to_libs + os.pathsep + os.environ["PATH"]


# Preserved ImportError messages when DPC++/SPMD backends fail to load.
# Used by _ensure_dpc_available() to surface actionable error messages.
# Only populated when the backend .so file exists but fails to import
# (e.g. missing SYCL runtime). Stays empty when the package is simply
# not installed — in that case "No module named X" is not informative.
_dpc_load_error: str = ""
_spmd_load_error: str = ""

_dpc_file_present = _backend_binary_present("_onedal_py_dpc")
_spmd_file_present = _backend_binary_present("_onedal_py_spmd_dpc")

try:
    # use dpc backend if available
    import onedal._onedal_py_dpc

    _dpc_backend = Backend(onedal._onedal_py_dpc, is_dpc=True, is_spmd=False)

    _host_backend = None
except ImportError as _dpc_import_err:
    # fall back to host backend; preserve reason only when the .so exists
    # (file-not-found ImportError is not actionable for end users)
    _dpc_backend = None
    if _dpc_file_present:
        _dpc_load_error = str(_dpc_import_err)

    import onedal._onedal_py_host

    _host_backend = Backend(onedal._onedal_py_host, is_dpc=False, is_spmd=False)

try:
    # also load spmd backend if available
    import onedal._onedal_py_spmd_dpc

    _spmd_backend = Backend(onedal._onedal_py_spmd_dpc, is_dpc=True, is_spmd=True)
except ImportError as _spmd_import_err:
    _spmd_backend = None
    if _spmd_file_present:
        _spmd_load_error = str(_spmd_import_err)

# if/elif/else layout required for pylint to realize _default_backend cannot be None
if _dpc_backend is not None:
    _default_backend = _dpc_backend
elif _host_backend is not None:
    _default_backend = _host_backend
else:
    raise ImportError("No oneDAL backend available")


def _ensure_dpc_available(require_spmd: bool = False) -> None:
    """Raise a user-actionable RuntimeError if the required DPC++/SPMD backend is unavailable.

    This function should be called when a SYCL queue is present but the
    corresponding backend was not loaded. It always raises when the backend
    is ``None`` (whether due to a load error or because the package was never
    installed), and includes the original ``ImportError`` reason when available.

    Parameters
    ----------
    require_spmd : bool, default=False
        If True, check the SPMD backend; otherwise check the DPC++ backend.

    Raises
    ------
    RuntimeError
        Always raised when the requested backend is unavailable.
        Includes the original ImportError reason (if captured) and
        install instructions.

    Notes
    -----
    Backend availability is determined at module import time. If the GPU
    package is installed after the interpreter has started, Python must be
    restarted for the change to take effect.
    """
    backend = _spmd_backend if require_spmd else _dpc_backend
    if backend is not None:
        return  # backend is available, nothing to do
    error_msg = _spmd_load_error if require_spmd else _dpc_load_error
    backend_label = "SPMD" if require_spmd else "DPC++"
    reason = f"\n  Reason: {error_msg}" if error_msg else ""
    raise RuntimeError(
        f"oneDAL GPU/{backend_label} support is not available "
        f"in the current installation.{reason}\n"
        "  To enable SYCL/GPU acceleration, install the GPU extras:\n"
        "    pip install scikit-learn-intelex-gpu\n"
        "  or via conda:\n"
        "    conda install scikit-learn-intelex-gpu -c "
        "https://software.repos.intel.com/python/conda"
    )


# Core modules to export
__all__ = [
    "_ensure_dpc_available",
    "_host_backend",
    "_default_backend",
    "_dpc_backend",
    "_spmd_backend",
    "covariance",
    "decomposition",
    "dummy",
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

# ===============================================================================
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
# ===============================================================================

from onedal.utils._third_party import dpctl_available

try:
    import dpnp

    dpnp_available = True
except ImportError:
    dpnp_available = False


if dpctl_available:
    import dpctl
    from dpctl.tensor import usm_ndarray

    def _get_sycl_queue(syclobj):
        if hasattr(syclobj, "_get_capsule"):
            return dpctl.SyclQueue(syclobj._get_capsule())
        else:
            return dpctl.SyclQueue(syclobj)

    def _assert_tensor_attr(actual, desired, order):
        """Check attributes of two given USM tensors."""
        is_usm_tensor = (
            lambda x: dpnp_available
            and isinstance(x, dpnp.ndarray)
            or isinstance(x, usm_ndarray)
        )
        assert is_usm_tensor(actual)
        assert is_usm_tensor(desired)
        # dpctl.tensor is the dpnp.ndarrays's core tensor structure along
        # with advanced device management. Convert dpnp to dpctl.tensor with zero copy.
        get_tensor = lambda x: (
            x.get_array() if dpnp_available and isinstance(x, dpnp.ndarray) else x
        )
        # Now DPCtl tensors
        actual = get_tensor(actual)
        desired = get_tensor(desired)

        assert actual.shape == desired.shape
        assert actual.strides == desired.strides
        assert actual.dtype == desired.dtype
        if order == "F":
            assert actual.flags.f_contiguous
            assert desired.flags.f_contiguous
            assert actual.flags.f_contiguous == desired.flags.f_contiguous
        else:
            assert actual.flags.c_contiguous
            assert desired.flags.c_contiguous
            assert actual.flags.c_contiguous == desired.flags.c_contiguous
        assert actual.flags == desired.flags
        assert actual.sycl_queue == desired.sycl_queue
        # TODO:
        # check better way to check usm ptrs.
        assert actual.usm_data._pointer == desired.usm_data._pointer

    def _assert_sua_iface_fields(
        actual, desired, skip_syclobj=False, skip_data_0=False, skip_data_1=False
    ):
        """Check attributes of two given reprsesentations of
        USM allocations `__sycl_usm_array_interface__`.

        For full documentation about `__sycl_usm_array_interface__` refer
        https://intelpython.github.io/dpctl/latest/api_reference/dpctl/sycl_usm_array_interface.html.

        Parameters
        ----------
        actual : dict, __sycl_usm_array_interface__
        desired : dict, __sycl_usm_array_interface__
        skip_syclobj : bool, default=False
            If True, check for __sycl_usm_array_interface__["syclobj"]
            will be skipped.
        skip_data_0 : bool, default=False
            If True, check for __sycl_usm_array_interface__["data"][0]
            will be skipped.
        skip_data_1 : bool, default=False
            If True, check for __sycl_usm_array_interface__["data"][1]
            will be skipped.
        """
        assert hasattr(actual, "__sycl_usm_array_interface__")
        assert hasattr(desired, "__sycl_usm_array_interface__")
        actual_sua_iface = actual.__sycl_usm_array_interface__
        desired_sua_iface = desired.__sycl_usm_array_interface__
        # data: A 2-tuple whose first element is a Python integer encoding
        # USM pointer value. The second entry in the tuple is a read-only flag
        # (True means the data area is read-only).
        if not skip_data_0:
            assert actual_sua_iface["data"][0] == desired_sua_iface["data"][0]
        if not skip_data_1:
            assert actual_sua_iface["data"][1] == desired_sua_iface["data"][1]
        # shape: a tuple of integers describing dimensions of an N-dimensional array.
        # Reformatting shapes for check cases (r,) vs (r,1). Contiguous flattened array
        # shape (r,) becoming (r,1) just for the check, since oneDAL supports only (r,1)
        # for 1-D arrays. In code after from_table conversion for 1-D expected outputs
        # xp.ravel or reshape(-1) is used.
        get_shape_if_1d = lambda shape: (shape[0], 1) if len(shape) == 1 else shape
        actual_shape = get_shape_if_1d(actual_sua_iface["shape"])
        desired_shape = get_shape_if_1d(desired_sua_iface["shape"])
        assert actual_shape == desired_shape
        # strides: An optional tuple of integers describing number of array elements
        # needed to jump to the next array element in the corresponding dimensions.
        if not actual_sua_iface["strides"] and not desired_sua_iface["strides"]:
            # None to indicate a C-style contiguous 1D array.
            # onedal4py constructs __sycl_usm_array_interface__["strides"] with
            # real values.
            assert actual_sua_iface["strides"] == desired_sua_iface["strides"]
        # versions: Version of the interface.
        assert actual_sua_iface["version"] == desired_sua_iface["version"]
        # typestr: a string encoding elemental data type of the array.
        assert actual_sua_iface["typestr"] == desired_sua_iface["typestr"]
        # syclobj: Python object from which SYCL context to which represented USM
        # allocation is bound.
        if not skip_syclobj and dpctl_available:
            actual_sycl_queue = _get_sycl_queue(actual_sua_iface["syclobj"])
            desired_sycl_queue = _get_sycl_queue(desired_sua_iface["syclobj"])
            assert actual_sycl_queue == desired_sycl_queue

# ==============================================================================
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
# ==============================================================================

import logging

import numpy as np
import pytest
import sklearn

import onedal
import sklearnex
from onedal.tests.utils._device_selection import is_dpctl_device_available


def test_get_config_contains_sklearn_params():
    skex_config = sklearnex.get_config()
    sk_config = sklearn.get_config()

    assert all(value in skex_config.keys() for value in sk_config.keys())


def test_set_config_works():
    """Test validates that the config settings were applied correctly by
    set_config.
    """
    # This retrieves the current configuration settings
    # from sklearnex to restore them later.
    default_config = sklearnex.get_config()

    # These variables define the new configuration settings
    # that will be tested.
    assume_finite = True
    target_offload = "cpu:0"
    allow_fallback_to_host = True
    allow_sklearn_after_onedal = False

    sklearnex.set_config(
        assume_finite=assume_finite,
        target_offload=target_offload,
        allow_fallback_to_host=allow_fallback_to_host,
        allow_sklearn_after_onedal=allow_sklearn_after_onedal,
    )

    config = sklearnex.get_config()
    onedal_config = onedal._config._get_config()
    # Any assert in test_set_config_works will leave the default config in place.
    # This is an undesired behavior. Using a try finally statement will guarantee
    # the use of set_config in the case of a failure.
    try:
        # These assertions check if the configuration was set correctly.
        # If any assertion fails, it will raise an error.
        assert config["target_offload"] == target_offload
        assert config["allow_fallback_to_host"] == allow_fallback_to_host
        assert config["allow_sklearn_after_onedal"] == allow_sklearn_after_onedal
        assert config["assume_finite"] == assume_finite
        assert onedal_config["target_offload"] == target_offload
        assert onedal_config["allow_fallback_to_host"] == allow_fallback_to_host
    finally:
        # This ensures that the original configuration is restored, regardless of
        # whether the assertions pass or fail.
        sklearnex.set_config(**default_config)


def test_config_context_works():
    """Test validates that the config settings were applied correctly
    by config context manager.
    """
    from sklearnex import config_context, get_config

    default_config = get_config()
    onedal_default_config = onedal._config._get_config()

    # These variables define the new configuration settings
    # that will be tested.
    assume_finite = True
    target_offload = "cpu:0"
    allow_fallback_to_host = True
    allow_sklearn_after_onedal = False

    # Nested context manager applies the new configuration settings.
    # Each config_context temporarily sets a specific configuration,
    # allowing for a clean and isolated testing environment.
    with config_context(assume_finite=assume_finite):
        with config_context(target_offload=target_offload):
            with config_context(allow_fallback_to_host=allow_fallback_to_host):
                with config_context(
                    allow_sklearn_after_onedal=allow_sklearn_after_onedal
                ):
                    config = sklearnex.get_config()
                    onedal_config = onedal._config._get_config()

    assert config["target_offload"] == target_offload
    assert config["allow_fallback_to_host"] == allow_fallback_to_host
    assert config["allow_sklearn_after_onedal"] == allow_sklearn_after_onedal
    assert config["assume_finite"] == assume_finite
    assert onedal_config["target_offload"] == target_offload
    assert onedal_config["allow_fallback_to_host"] == allow_fallback_to_host

    # Check that out of the config context manager default settings are
    # remaining.
    default_config_after_cc = get_config()
    onedal_default_config_after_cc = onedal._config._get_config()
    for param in [
        "target_offload",
        "allow_fallback_to_host",
        "allow_sklearn_after_onedal",
        "assume_finite",
    ]:
        assert default_config_after_cc[param] == default_config[param]

    for param in [
        "target_offload",
        "allow_fallback_to_host",
    ]:
        assert onedal_default_config_after_cc[param] == onedal_default_config[param]


@pytest.mark.skipif(
    onedal._default_backend.is_dpc, reason="requires host default backend"
)
@pytest.mark.parametrize("target", ["auto", "cpu", "cpu:0", "gpu", 3])
def test_host_backend_target_offload(target):
    from sklearnex.neighbors import NearestNeighbors

    err_msg = (
        r"device use via \`target_offload\` is only supported with the DPC\+\+ backend"
    )

    est = NearestNeighbors()
    if target != "auto":
        with pytest.raises(ValueError, match=err_msg):
            with sklearnex.config_context(target_offload=target):
                est.fit(np.eye(5, 8))
    else:
        with sklearnex.config_context(target_offload=target):
            est.fit(np.eye(5, 8))


@pytest.mark.skipif(
    not is_dpctl_device_available(["gpu"]), reason="Requires a gpu for fallback testing"
)
def test_fallback_to_host(caplog):
    # force a fallback to cpu with direct use of dispatch and PatchingConditionsChain
    # it should complete with allow_fallback_to_host. The queue should be preserved
    # and properly used in the second round on gpu
    from onedal.utils import _sycl_queue_manager as QM
    from sklearnex._device_offload import dispatch
    from sklearnex._utils import PatchingConditionsChain

    class _Estimator:
        def _onedal_gpu_supported(self, method_name, *data):
            patching_status = PatchingConditionsChain("")
            patching_status.and_condition(data[0] == "gpu", "")
            return patching_status

        def _onedal_cpu_supported(self, method_name, *data):
            patching_status = PatchingConditionsChain("")
            return patching_status

        def _onedal_test(self, *args, queue=None):
            if args[0] == "cpu":
                assert (
                    queue is None
                    and QM.__global_queue == QM.__fallback_queue
                    and QM.get_global_queue() is None
                )
            elif args[0] == "gpu":
                assert queue is not None and QM.get_global_queue() is not None

    start = 0
    est = _Estimator()

    # set a queue which should persist
    with (
        caplog.at_level(logging.DEBUG, logger="sklearnex"),
        sklearnex.config_context(target_offload="gpu"),
    ):
        # True == with cpu (eventually), False == with gpu
        for fallback in [True, False]:
            with sklearnex.config_context(allow_fallback_to_host=fallback):
                dispatch(
                    est,
                    "test",
                    {"onedal": est._onedal_test, "sklearn": None},
                    "cpu" if fallback else "gpu",
                )

            # verify that the target_offload has not changed
            assert sklearnex.get_config()["target_offload"] == "gpu"
            assert (
                f": running accelerated version on {'CPU' if fallback else 'GPU'}"
                in caplog.messages[start:]
            )
            start = len(caplog.messages)

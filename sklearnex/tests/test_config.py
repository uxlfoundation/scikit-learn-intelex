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

import pytest
import scipy.sparse as sp
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
    not is_dpctl_device_available("gpu"), reason="Requires a gpu for fallback testing"
)
def test_fallback_to_host(caplog):
    # force a fallback to cpu using sparse data and sample weights in BasicStatistics
    # it should complete with allow_fallback_to_host. The queue should be preserved
    # and properly used in the second round on gpu
    from sklearnex.basic_statistics import BasicStatistics

    est = BasicStatistics()
    # set a queue which should persist
    start = 0
    with (
        caplog.at_level(logging.WARNING, logger="sklearnex"),
        config_context(target_offload="gpu"),
    ):
        # True == with cpu (eventually), False == with gpu
        for fallback, data in [[True, sp.eye(5, 8)], [False, np.eye(5, 8)]]:
            with config_context(allow_fallback_to_host=fallback):
                est.fit(data, sample_weight=data)

            assert (
                f"running accelerated version on {'CPU' if fallback else 'GPU'}"
                in caplog.records[start:]
            ), "".join(caplog.records)
            start = len(caplog.records)

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

import logging
import os
import sys
import warnings

from daal4py.sklearn._utils import (
    PatchingConditionsChain as daal4py_PatchingConditionsChain,
)
from daal4py.sklearn._utils import daal_check_version


class PatchingConditionsChain(daal4py_PatchingConditionsChain):
    def get_status(self):
        return self.patching_is_enabled

    def write_log(self, queue=None, transferred_to_host=True):
        if self.patching_is_enabled:
            self.logger.info(
                f"{self.scope_name}: {get_patch_message('onedal', queue=queue, transferred_to_host=transferred_to_host)}"
            )
        else:
            self.logger.debug(
                f"{self.scope_name}: debugging for the patch is enabled to track"
                " the usage of Intel® oneAPI Data Analytics Library (oneDAL)"
            )
            for message in self.messages:
                self.logger.debug(
                    f"{self.scope_name}: patching failed with cause - {message}"
                )
            self.logger.info(
                f"{self.scope_name}: {get_patch_message('sklearn', transferred_to_host=transferred_to_host)}"
            )


def set_sklearn_ex_verbose():
    log_level = os.environ.get("SKLEARNEX_VERBOSE")

    logger = logging.getLogger("sklearnex")
    logging_channel = logging.StreamHandler()
    logging_formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    logging_channel.setFormatter(logging_formatter)
    logger.addHandler(logging_channel)

    try:
        if log_level is not None:
            logger.setLevel(log_level)
    except Exception:
        warnings.warn(
            'Unknown level "{}" for logging.\n'
            'Please, use one of "CRITICAL", "ERROR", '
            '"WARNING", "INFO", "DEBUG".'.format(log_level)
        )


def get_patch_message(s, queue=None, transferred_to_host=True):
    if s == "onedal":
        message = "running accelerated version on "
        if queue is not None:
            if queue.sycl_device.is_gpu:
                message += "GPU"
            elif queue.sycl_device.is_cpu:
                message += "CPU"
            else:
                raise RuntimeError("Unsupported device")
        else:
            message += "CPU"
    elif s == "sklearn":
        message = "fallback to original Scikit-learn"
    elif s == "sklearn_after_onedal":
        message = "failed to run accelerated version, fallback to original Scikit-learn"
    else:
        raise ValueError(
            f"Invalid input - expected one of 'onedal','sklearn',"
            f" 'sklearn_after_onedal', got {s}"
        )
    if transferred_to_host:
        message += (
            ". All input data transferred to host for further backend computations."
        )
    return message


def get_usm_data_message(is_copied=False):
    message = ""
    if is_copied:
        message = ". All USM inputs are being copied to HOST for further computations."
    return message


def register_hyperparameters(hyperparameters_map):
    """Decorator for hyperparameters support in estimator class.
    Adds `get_hyperparameters` method to class.
    """

    def wrap_class(estimator_class):
        def get_hyperparameters(self, op):
            return hyperparameters_map[op]

        estimator_class.get_hyperparameters = get_hyperparameters
        return estimator_class

    return wrap_class

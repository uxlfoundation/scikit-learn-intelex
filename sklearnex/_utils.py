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
import re
import sys
import warnings
from abc import ABC

import sklearn

from daal4py.sklearn._utils import (
    PatchingConditionsChain as daal4py_PatchingConditionsChain,
)
from daal4py.sklearn._utils import (
    daal_check_version,
    sklearn_check_version,
)

# Note: if inheriting from '_HTMLDocumentationLinkMixin' here, it then doesn't matter
# the order of inheritance of classes for estimators when this is later subclassed,
# whereas if inheriting from something else, the subclass that inherits from this needs
# to be the first inherited class in estimators in order for it to take effect.
if sklearn_check_version("1.4"):
    from sklearn.utils._estimator_html_repr import _HTMLDocumentationLinkMixin

    BaseForHTMLDocLink = _HTMLDocumentationLinkMixin
else:
    BaseForHTMLDocLink = ABC


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
                " the usage of oneAPI Data Analytics Library (oneDAL)"
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


def get_sklearnex_version(rule):
    return daal_check_version(rule)


def register_hyperparameters(hyperparameters_map):
    """Decorator for hyperparameters support in estimator class.
    Adds `get_hyperparameters` method to class.
    """

    def decorator(cls):
        """Add `get_hyperparameters()` static method"""

        class StaticHyperparametersAccessor:
            """Like a @staticmethod, but additionally raises a Warning when called on an instance."""

            def __get__(self, instance, _):
                if instance is not None:
                    warnings.warn(
                        "Hyperparameters are static variables and can not be modified per instance."
                    )
                return self.get_hyperparameters

            def get_hyperparameters(self, op):
                return hyperparameters_map[op]

        cls.get_hyperparameters = StaticHyperparametersAccessor()
        return cls

    return decorator


def _add_inc_serialization_note(class_docstrings: str) -> str:
    """Adds a small note note about serialization for extension estimators that are incremental.
    The class docstrings should leave a placeholder '%incremental_serialization_note%' inside
    their docstrings, which will be replaced by this note.
    """
    # In python versions >=3.13, leading whitespace in docstrings defined through
    # static strings (but **not through other ways**) is automatically removed
    # from the final docstrings, while in earlier versions is kept.
    inc_serialization_note = """Note
----
Serializing instances of this class will trigger a forced finalization of calculations
when the inputs are in a sycl queue or when using GPUs. Since (internal method)
finalize_fit can't be dispatched without directly provided queue and the dispatching
policy can't be serialized, the computation is finalized during serialization call and
the policy is not saved in serialized data."""
    if sys.version_info.major == 3 and sys.version_info.minor <= 12:
        inc_serialization_note = re.sub(
            r"^", " " * 4, inc_serialization_note, flags=re.MULTILINE
        )
        inc_serialization_note = inc_serialization_note.strip()
    return class_docstrings.replace(
        r"%incremental_serialization_note%", inc_serialization_note
    )

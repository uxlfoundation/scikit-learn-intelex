# ===============================================================================
# Copyright 2023 Intel Corporation
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

import warnings
from functools import update_wrapper

from daal4py.sklearn._utils import sklearn_check_version

from .._config import config_context, get_config

# Replacement of _FuncWrapper is required to correctly propagate
# the scikit-learn-intelex configuration functions to the joblib workers.
if sklearn_check_version("1.7"):

    class _FuncWrapper:
        """Load the global configuration before calling the function."""

        def __init__(self, function):
            self.function = function
            update_wrapper(self, self.function)

        def with_config_and_warning_filters(self, config, warning_filters):
            self.config = config
            self.warning_filters = warning_filters
            return self

        def __call__(self, *args, **kwargs):
            config = getattr(self, "config", {})
            warning_filters = getattr(self, "warning_filters", [])
            if not config or not warning_filters:
                warnings.warn(
                    (
                        "`sklearn.utils.parallel.delayed` should be used with"
                        " `sklearn.utils.parallel.Parallel` to make it possible to"
                        " propagate the scikit-learn configuration of the current thread to"
                        " the joblib workers."
                    ),
                    UserWarning,
                )

            with config_context(**config), warnings.catch_warnings():
                warnings.filters = warning_filters
                return self.function(*args, **kwargs)

elif sklearn_check_version("1.2.1"):

    class _FuncWrapper:
        """Load the global configuration before calling the function."""

        def __init__(self, function):
            self.function = function
            update_wrapper(self, self.function)

        def with_config(self, config):
            self.config = config
            return self

        def __call__(self, *args, **kwargs):
            config = getattr(self, "config", None)
            if config is None:
                warnings.warn(
                    "`sklearn.utils.parallel.delayed` should be used with "
                    "`sklearn.utils.parallel.Parallel` to make it possible to propagate "
                    "the scikit-learn configuration of the current thread to the "
                    "joblib workers.",
                    UserWarning,
                )
                config = {}
            with config_context(**config):
                return self.function(*args, **kwargs)

else:

    class _FuncWrapper:
        """Load the global configuration before calling the function."""

        def __init__(self, function):
            self.function = function
            self.config = get_config()
            update_wrapper(self, self.function)

        def __call__(self, *args, **kwargs):
            with config_context(**self.config):
                return self.function(*args, **kwargs)

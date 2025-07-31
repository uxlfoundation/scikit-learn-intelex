# ==============================================================================
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
# ==============================================================================

import logging
from typing import Callable, Dict, Tuple
from warnings import warn

from daal4py.sklearn._utils import daal_check_version
from onedal import _default_backend as backend

if not daal_check_version((2024, "P", 0)):
    warn("Hyperparameters are supported in oneDAL starting from 2024.0.0 version.")
    hyperparameters_map = {}
else:
    _hparams_reserved_words = [
        "algorithm",
        "op",
        "setters",
        "getters",
        "backend",
        "is_default",
        "to_dict",
    ]

    class HyperParameters:
        """Class for simplified interaction with oneDAL hyperparameters.
        Overrides `__getattribute__` and `__setattr__` to utilize getters and setters
        of hyperparameter class from onedal backend.
        """

        def __init__(self, algorithm, op, setters, getters, backend):
            self.algorithm = algorithm
            self.op = op
            self.setters = setters
            self.getters = getters
            self.backend = backend
            self.is_default = True

        def __getattribute__(self, __name):
            if __name in _hparams_reserved_words:
                if __name == "backend":
                    # `backend` attribute accessed only for oneDAL kernel calls
                    logging.getLogger("sklearnex").debug(
                        "Using next hyperparameters for "
                        f"'{self.algorithm}.{self.op}': {self.to_dict()}"
                    )
                return super().__getattribute__(__name)
            elif __name in self.getters.keys():
                return self.getters[__name]()
            try:
                # try to return attribute from base class
                # required to read builtin attributes like __class__, __doc__, etc.
                # which are used in debuggers
                return super().__getattribute__(__name)
            except AttributeError:
                # raise an AttributeError with a hyperparameter-specific message
                # for easier debugging
                raise AttributeError(
                    f"Unknown attribute '{__name}' in "
                    f"'{self.algorithm}.{self.op}' hyperparameters"
                )

        def __setattr__(self, __name, __value):
            if __name in _hparams_reserved_words:
                super().__setattr__(__name, __value)
            elif __name in self.setters.keys():
                self.is_default = False
                self.setters[__name](__value)
            else:
                raise ValueError(
                    f"Unknown attribute '{__name}' in "
                    f"'{self.algorithm}.{self.op}' hyperparameters"
                )

        def to_dict(self):
            return {name: getter() for name, getter in self.getters.items()}

    def get_methods_with_prefix(obj, prefix):
        return {
            method.replace(prefix, ""): getattr(obj, method)
            for method in filter(lambda f: f.startswith(prefix), dir(obj))
        }

    hyperparameters_backend: Dict[Tuple[str, str], Callable] = {
        (
            "linear_regression",
            "train",
        ): lambda: backend.linear_model.regression.train_hyperparameters(),
        ("covariance", "compute"): lambda: backend.covariance.compute_hyperparameters(),
    }
    if daal_check_version((2024, "P", 300)):
        hyperparameters_backend[("decision_forest", "infer")] = (
            lambda: backend.decision_forest.infer_hyperparameters()
        )
    if daal_check_version((2025, "P", 700)):
        hyperparameters_backend[("pca", "train")] = (
            lambda: backend.decomposition.dim_reduction.train_hyperparameters()
        )

    hyperparameters_map: Dict[Tuple[str, str], HyperParameters] = {}

    for (algorithm, op), hyperparameters_lambda in hyperparameters_backend.items():
        hyperparameters = hyperparameters_lambda()
        setters = get_methods_with_prefix(hyperparameters, "set_")
        getters = get_methods_with_prefix(hyperparameters, "get_")

        if set(setters.keys()) != set(getters.keys()):
            raise ValueError(
                f"Setters and getters in '{algorithm}.{op}' "
                "hyperparameters wrapper do not correspond."
            )

        hyperparameters_map[(algorithm, op)] = HyperParameters(
            algorithm, op, setters, getters, hyperparameters
        )

    def get_hyperparameters_backend(algorithm, op):
        """Get hyperparameters for a specific algorithm and operation."""
        if (algorithm, op) in hyperparameters_backend:
            return hyperparameters_backend[(algorithm, op)]()
        else:
            raise ValueError(f"Hyperparameters for '{algorithm}.{op}' are not defined.")


def get_hyperparameters(algorithm: str, op: str) -> HyperParameters:
    return hyperparameters_map.get((algorithm, op), None)


def reset_hyperparameters(algorithm: str, op: str) -> None:
    new_hyperparameters_backend = get_hyperparameters_backend(algorithm, op)
    new_setters = get_methods_with_prefix(new_hyperparameters_backend, "set_")
    new_getters = get_methods_with_prefix(new_hyperparameters_backend, "get_")
    hyperparameters_map[(algorithm, op)] = HyperParameters(
        algorithm, op, new_setters, new_getters, new_hyperparameters_backend
    )

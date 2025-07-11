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

from daal4py.sklearn._utils import sklearn_check_version
from daal4py.sklearn.linear_model import ElasticNet, Lasso
from onedal._device_offload import support_input_format

from ..base import oneDALEstimator

# Note: `sklearnex.linear_model.ElasticNet` only has functional
# sycl GPU support. No GPU device will be offloaded.
ElasticNet.fit = support_input_format(ElasticNet.fit)
ElasticNet.predict = support_input_format(ElasticNet.predict)
ElasticNet.score = support_input_format(ElasticNet.score)
if sklearn_check_version("1.4"):
    ElasticNet._doc_link_module = "daal4py"
    ElasticNet._doc_link_url_param_generator = (
        oneDALEstimator._doc_link_url_param_generator
    )


# Note: `sklearnex.linear_model.Lasso` only has functional
# sycl GPU support. No GPU device will be offloaded.
Lasso.fit = support_input_format(Lasso.fit)
Lasso.predict = support_input_format(Lasso.predict)
Lasso.score = support_input_format(Lasso.score)
if sklearn_check_version("1.4"):
    Lasso._doc_link_module = "daal4py"
    Lasso._doc_link_url_param_generator = oneDALEstimator._doc_link_url_param_generator

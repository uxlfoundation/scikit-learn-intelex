/*******************************************************************************
* Copyright 2020 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

// Definitions/declarations, mapping cython names/types to DAAL's actual types

#ifndef _LOG_REG_MODEL_BUILDER_INCLUDED_
#define _LOG_REG_MODEL_BUILDER_INCLUDED_

#include <daal.h>

typedef daal::algorithms::logistic_regression::ModelBuilder<DAAL_ALGORITHM_FP_TYPE> c_logistic_regression_model_builder;

template <typename modelFPType = DAAL_ALGORITHM_FP_TYPE>
static daal::algorithms::logistic_regression::ModelPtr * get_logistic_regression_model_builder_model(daal::algorithms::logistic_regression::ModelBuilder<modelFPType> * obj_)
{
    return RAW<daal::algorithms::logistic_regression::ModelPtr>()(obj_->getModel());
}

static daal::data_management::NumericTablePtr getTable(const data_or_file & t)
{
    return get_table(t);
}

#endif // _LOG_REG_MODEL_BUILDER_INCLUDED_

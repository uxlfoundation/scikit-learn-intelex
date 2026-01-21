/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef _DF_MODEL_BUILDER_INCLUDED_
#define _DF_MODEL_BUILDER_INCLUDED_

#include <daal.h>

typedef daal::algorithms::decision_forest::classification::ModelBuilder c_df_classification_model_builder;

typedef c_df_classification_model_builder::NodeId c_df_clf_node_id;
typedef c_df_classification_model_builder::TreeId c_df_clf_tree_id;

#define c_df_clf_no_parent c_df_classification_model_builder::noParent

static daal::algorithms::decision_forest::classification::ModelPtr * get_decision_forest_classification_model_builder_model(daal::algorithms::decision_forest::classification::ModelBuilder * obj_)
{
    return RAW<daal::algorithms::decision_forest::classification::ModelPtr>()(obj_->getModel());
}

#endif // _DF_MODEL_BUILDER_INCLUDED_

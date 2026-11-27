#===============================================================================
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
#===============================================================================

# We expose DAAL's directly through Cython
# The model builder object is retrieved through calling model_builder.
# We will extend this once we know how other model builders will work in DAAL

cdef extern from "df_model_builder.h":
    ctypedef size_t c_df_clf_node_id
    ctypedef size_t c_df_clf_tree_id

    cdef size_t c_df_clf_no_parent

    cdef cppclass c_df_classification_model_builder:
        c_df_classification_model_builder(size_t nClasses, size_t nTrees) except +
        c_df_clf_tree_id createTree(size_t nNodes)
        c_df_clf_node_id addLeafNode(c_df_clf_tree_id tree_Id, c_df_clf_node_id parent_id, size_t position, size_t classLabel)
        c_df_clf_node_id addSplitNode(c_df_clf_tree_id tree_Id, c_df_clf_node_id parent_id, size_t position, size_t featureIndex, double featureValue)
        void setNFeatures(size_t n_features)

    cdef decision_forest_classification_ModelPtr * get_decision_forest_classification_model_builder_model(c_df_classification_model_builder *)


cdef class decision_forest_classification_model_builder:
    '''
    Model Builder for decision forest.
    '''
    cdef c_df_classification_model_builder * c_ptr

    def __cinit__(self, size_t nClasses, size_t nTrees):
        self.c_ptr = new c_df_classification_model_builder(nClasses, nTrees)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t n_nodes):
        '''
        Create one tree of the forest

        :param size_t n_nodes: number of nodes in created tree
        :rtype: tree identifier
        '''
        return self.c_ptr.createTree(n_nodes)

    def add_leaf(self, c_df_clf_tree_id tree_id, size_t classLabel, c_df_clf_node_id parent_id=c_df_clf_no_parent, size_t position=0):
        '''
        Create Leaf node

        :param tree-handle tree_id: tree to which new node is added
        :param size_t classLabel: class-label
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in root node [optional, default: 0]
        :rtype: node identifier
        '''
        return self.c_ptr.addLeafNode(tree_id, parent_id, position, classLabel)

    def add_split(self, c_df_clf_tree_id tree_id, size_t feature_index, double feature_value, c_df_clf_node_id parent_id=c_df_clf_no_parent, size_t position=0):
        '''
        Create Split node

        :param tree-handle tree_id: tree to which node is added
        :param size_t feature_index: feature index for spliting
        :param double feature_value: feature value for spliting
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :rtype: node identifier
        '''
        return self.c_ptr.addSplitNode(tree_id, parent_id, position, feature_index, feature_value)

    def set_n_features(self, size_t n_features):
        '''
        Set number of features

        :param size_t n_features: number of features
        '''
        return self.c_ptr.setNFeatures(n_features)

    def model(self):
        '''
        Get built model

        :rtype: decision_forest_classification_model
        '''
        cdef decision_forest_classification_model res = decision_forest_classification_model.__new__(decision_forest_classification_model)
        res.c_ptr = get_decision_forest_classification_model_builder_model(self.c_ptr)
        return res

def decision_forest_clf_model_builder(nTrees=1, nClasses=2):
    '''
    Model builder for decision forest classification

    :param size_t nTrees: Number of trees in the decision forest
    :param size_t nClasses: Number of classes in model
    '''
    return decision_forest_classification_model_builder(nClasses, nTrees)

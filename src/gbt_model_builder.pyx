#===============================================================================
# Copyright 2020 Intel Corporation
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

cdef extern from "gbt_model_builder.h":
    ctypedef size_t c_gbt_clf_node_id
    ctypedef size_t c_gbt_clf_tree_id
    ctypedef size_t c_gbt_reg_node_id
    ctypedef size_t c_gbt_reg_tree_id

    cdef size_t c_gbt_clf_no_parent
    cdef size_t c_gbt_reg_no_parent

    cdef gbt_classification_ModelPtr * get_gbt_classification_model_builder_model(c_gbt_classification_model_builder *, double base_score)
    cdef gbt_regression_ModelPtr * get_gbt_regression_model_builder_model(c_gbt_regression_model_builder *, double base_score)

    cdef cppclass c_gbt_classification_model_builder:
        c_gbt_classification_model_builder(size_t nFeatures, size_t nIterations, size_t nClasses) except +
        c_gbt_clf_tree_id createTree(size_t nNodes, size_t classLabel)
        c_gbt_clf_node_id addLeafNode(c_gbt_clf_tree_id treeId, c_gbt_clf_node_id parentId, size_t position, double response, double cover)

    cdef cppclass c_gbt_regression_model_builder:
        c_gbt_regression_model_builder(size_t nFeatures, size_t nIterations) except +
        c_gbt_reg_tree_id createTree(size_t nNodes)
        c_gbt_reg_node_id addLeafNode(c_gbt_reg_tree_id treeId, c_gbt_reg_node_id parentId, size_t position, double response, double cover)

    cdef c_gbt_clf_node_id clfAddSplitNodeWrapper(c_gbt_classification_model_builder * c_ptr, c_gbt_clf_tree_id treeId, c_gbt_clf_node_id parentId, size_t position, size_t featureIndex, double featureValue, int defaultLeft, double cover)
    cdef c_gbt_reg_node_id regAddSplitNodeWrapper(c_gbt_regression_model_builder     * c_ptr, c_gbt_reg_tree_id treeId, c_gbt_reg_node_id parentId, size_t position, size_t featureIndex, double featureValue, int defaultLeft, double cover)

    cdef c_gbt_clf_node_id clfAddLeafNodeWrapper(c_gbt_classification_model_builder * c_ptr, c_gbt_clf_tree_id treeId, c_gbt_clf_node_id parentId, size_t position, double response, double cover)
    cdef c_gbt_clf_node_id regAddLeafNodeWrapper(c_gbt_regression_model_builder     * c_ptr, c_gbt_reg_tree_id treeId, c_gbt_reg_node_id parentId, size_t position, double response, double cover)

cdef class gbt_classification_model_builder:
    '''
    Model Builder for gradient boosted trees.
    '''
    cdef c_gbt_classification_model_builder * c_ptr

    def __cinit__(self, size_t n_features, size_t n_iterations, size_t n_classes):
        self.c_ptr = new c_gbt_classification_model_builder(n_features, n_iterations, n_classes)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t n_nodes, size_t class_label):
        '''
        Create certain tree in the gradient boosted trees classification model for certain class

        :param size_t n_nodes: number of nodes in created tree
        :param size_t class_label: label of class for which tree is created. classLabel bellows interval from 0 to (nClasses - 1)
        :rtype: tree identifier
        '''
        return self.c_ptr.createTree(n_nodes, class_label)

    def add_leaf(self, c_gbt_clf_tree_id tree_id, double response, double cover, c_gbt_clf_node_id parent_id=c_gbt_clf_no_parent, size_t position=0):
        '''
        Create Leaf node and add it to certain tree

        :param tree-handle tree_id: tree to which new node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param double response: response value for leaf node to be predicted
        :param double cover: cover (sum_hess) of the leaf node
        :rtype: node identifier
        '''
        return clfAddLeafNodeWrapper(self.c_ptr, tree_id, parent_id, position, response, cover)

    def add_split(self, c_gbt_clf_tree_id tree_id, size_t feature_index, double feature_value, int default_left, double cover, c_gbt_clf_node_id parent_id=c_gbt_clf_no_parent, size_t position=0):
        '''
        Create Split node and add it to certain tree.

        :param tree-handle tree_id: tree to which node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param size_t feature_index: feature index for splitting
        :param double feature_value: feature value for splitting
        :param int default_left: default behaviour in case of missing value
        :param double cover: cover (sum_hess) of the solit node
        :rtype: node identifier
        '''
        return clfAddSplitNodeWrapper(self.c_ptr, tree_id, parent_id, position, feature_index, feature_value, default_left, cover)

    def model(self, base_score):
        '''
        Get built model

        :param double base_score: global prediction bias (used e.g. in XGBoost)
        :rtype: gbt_classification_model
        '''
        cdef gbt_classification_model res = gbt_classification_model.__new__(gbt_classification_model)
        res.c_ptr = get_gbt_classification_model_builder_model(self.c_ptr, base_score or 0.0)
        return res


cdef class gbt_regression_model_builder:
    '''
    Model Builder for gradient boosted trees.
    '''
    cdef c_gbt_regression_model_builder * c_ptr

    def __cinit__(self, size_t n_features, size_t n_iterations):
        self.c_ptr = new c_gbt_regression_model_builder(n_features, n_iterations)

    def __dealloc__(self):
        del self.c_ptr

    def create_tree(self, size_t n_nodes):
        '''
        Create certain tree in the gradient boosted trees regression model

        :param size_t n_nodes: number of nodes in created tree
        :rtype: tree identifier
        '''
        return self.c_ptr.createTree(n_nodes)

    def add_leaf(self, c_gbt_reg_tree_id tree_id, double response, double cover, c_gbt_reg_node_id parent_id=c_gbt_reg_no_parent, size_t position=0):
        '''
        Create Leaf node and add it to certain tree

        :param tree-handle tree_id: tree to which new node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param double response: response value for leaf node to be predicted
        :param double cover: cover (sum_hess) of the leaf node
        :rtype: node identifier
        '''
        return regAddLeafNodeWrapper(self.c_ptr, tree_id, parent_id, position, response, cover)

    def add_split(self, c_gbt_reg_tree_id tree_id, size_t feature_index, double feature_value, int default_left, double cover, c_gbt_reg_node_id parent_id=c_gbt_reg_no_parent, size_t position=0):
        '''
        Create Split node and add it to certain tree.

        :param tree-handle tree_id: tree to which node is added
        :param node-handle parent_id: parent node to which new node is added (use noParent for root node)
        :param size_t position: position in parent (e.g. 0 for left and 1 for right child in a binary tree)
        :param size_t feature_index: feature index for splitting
        :param double feature_value: feature value for splitting
        :param double cover: cover (sum_hess) of the split node
        :param int default_left: default behaviour in case of missing value
        :rtype: node identifier
        '''
        return regAddSplitNodeWrapper(self.c_ptr, tree_id, parent_id, position, feature_index, feature_value, default_left, cover)

    def model(self, base_score):
        '''
        Get built model

        :param double base_score: global prediction bias (used e.g. in XGBoost)
        :rtype: gbt_regression_model
        '''
        cdef gbt_regression_model res = gbt_regression_model.__new__(gbt_regression_model)
        res.c_ptr = get_gbt_regression_model_builder_model(self.c_ptr, base_score or 0.0)
        return res


def gbt_clf_model_builder(n_features, n_iterations, n_classes = 2):
    '''
    Model builder for gradient boosted trees classification

    :param size_t n_features: Number of features in training data
    :param size_t n_iterations: Number of trees in model for each class
    :param size_t n_classes: Number of classes in model
    '''
    return gbt_classification_model_builder(n_features, n_iterations, n_classes)


def gbt_reg_model_builder(n_features, n_iterations):
    '''
    Model builder for gradient boosted trees regression

    :param size_t n_features: Number of features in training data
    :param size_t n_iterations: Number of trees in the model
    '''
    return gbt_regression_model_builder(n_features, n_iterations)

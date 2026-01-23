# ==============================================================================
# Copyright 2014 Intel Corporation
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

# daal4py Decision Forest Regression Tree Traversal example

import math

from decision_forest_regression_default_dense import main as df_regression

import daal4py as d4p


def printTree(nodes, values):
    def printNodes(node_id, nodes, values, level):
        node = nodes[node_id]
        value = values[node_id]
        if not math.isnan(node["threshold"]):
            print(
                " " * level + "Level " + str(level) + ": Feature ="
                " " + str(node["feature"]) + ", Threshold = " + str(node["threshold"])
            )
        else:
            print(
                " " * level + "Level " + str(level) + ", Value ="
                " " + str(value).replace(" ", "")
            )
        if node["left_child"] != -1:
            printNodes(node["left_child"], nodes, values, level + 1)
        if node["right_child"] != -1:
            printNodes(node["right_child"], nodes, values, level + 1)

    printNodes(0, nodes, values, 0)


if __name__ == "__main__":
    # First get our result and model
    train_result, _, _ = df_regression()
    # Retrieve and print all trees; encoded as in sklearn.ensamble.tree_.Tree
    for treeId in range(train_result.model.NumberOfTrees):
        treeState = d4p.getTreeState(train_result.model, treeId)
        printTree(treeState.node_ar, treeState.value_ar)
    print("Traversed {} trees.".format(train_result.model.NumberOfTrees))
    print("All looks good!")

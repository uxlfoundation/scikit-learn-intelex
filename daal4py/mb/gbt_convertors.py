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

import json
import warnings
from collections import deque
from copy import deepcopy
from tempfile import NamedTemporaryFile
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

from .. import gbt_clf_model_builder, gbt_reg_model_builder


class CatBoostNode:
    def __init__(
        self,
        split: Optional[Dict] = None,
        value: Optional[List[float]] = None,
        right: Optional[int] = None,
        left: Optional[float] = None,
        cover: Optional[float] = None,
    ) -> None:
        self.split = split
        self.value = value
        self.right = right
        self.left = left
        self.cover = cover


class CatBoostModelData:
    """Wrapper around the CatBoost model dump for easier access to properties"""

    def __init__(self, data):
        self.__data = data

    @property
    def n_features(self):
        return len(self.__data["features_info"]["float_features"])

    @property
    def grow_policy(self):
        return self.__data["model_info"]["params"]["tree_learner_options"]["grow_policy"]

    @property
    def oblivious_trees(self):
        return self.__data["oblivious_trees"]

    @property
    def trees(self):
        return self.__data["trees"]

    @property
    def n_classes(self):
        """Number of classes, returns -1 if it's not a classification model"""
        if "class_params" in self.__data["model_info"]:
            return len(self.__data["model_info"]["class_params"]["class_to_label"])
        return -1

    @property
    def is_classification(self):
        return "class_params" in self.__data["model_info"]

    @property
    def has_categorical_features(self):
        return "categorical_features" in self.__data["features_info"]

    @property
    def is_symmetric_tree(self):
        return self.grow_policy == "SymmetricTree"

    @property
    def float_features(self):
        return self.__data["features_info"]["float_features"]

    @property
    def n_iterations(self):
        if self.is_symmetric_tree:
            return len(self.oblivious_trees)
        else:
            return len(self.trees)

    @property
    def scale(self):
        return self.__data["scale_and_bias"][0]

    @property
    def default_left(self):
        dpo = self.__data["model_info"]["params"]["data_processing_options"]
        nan_mode = dpo["float_features_binarization"]["nan_mode"]
        return int(nan_mode.lower() == "min")


class Node:
    """Helper class holding Tree Node information"""

    def __init__(
        self,
        cover: float,
        is_leaf: bool,
        default_left: bool,
        feature: int,
        value: float,
        n_children: int = 0,
        left_child: "Optional[Node]" = None,
        right_child: "Optional[Node]" = None,
        parent_id: Optional[int] = -1,
        position: Optional[int] = -1,
    ) -> None:
        self.cover = cover
        self.is_leaf = is_leaf
        self.default_left = default_left
        self.__feature = feature
        self.value = value
        self.n_children = n_children
        self.left_child = left_child
        self.right_child = right_child
        self.parent_id = parent_id
        self.position = position

    @staticmethod
    def from_xgb_dict(
        input_dict: Dict[str, Any], feature_names_to_indices: dict[str, int]
    ) -> "Node":
        if "children" in input_dict:
            left_child = Node.from_xgb_dict(
                input_dict["children"][0], feature_names_to_indices
            )
            right_child = Node.from_xgb_dict(
                input_dict["children"][1], feature_names_to_indices
            )
            n_children = 2 + left_child.n_children + right_child.n_children
        else:
            left_child = None
            right_child = None
            n_children = 0
        is_leaf = "leaf" in input_dict
        default_left = "yes" in input_dict and input_dict["yes"] == input_dict["missing"]
        feature = input_dict.get("split")
        if feature:
            feature = feature_names_to_indices[feature]
        return Node(
            cover=input_dict["cover"],
            is_leaf=is_leaf,
            default_left=default_left,
            feature=feature,
            value=input_dict["leaf"] if is_leaf else input_dict["split_condition"],
            n_children=n_children,
            left_child=left_child,
            right_child=right_child,
        )

    @staticmethod
    def from_lightgbm_dict(input_dict: Dict[str, Any]) -> "Node":
        if "tree_structure" in input_dict:
            tree = input_dict["tree_structure"]
        else:
            tree = input_dict

        n_children = 0
        if "left_child" in tree:
            left_child = Node.from_lightgbm_dict(tree["left_child"])
            n_children += 1 + left_child.n_children
        else:
            left_child = None
        if "right_child" in tree:
            right_child = Node.from_lightgbm_dict(tree["right_child"])
            n_children += 1 + right_child.n_children
        else:
            right_child = None

        is_leaf = "leaf_value" in tree
        # get cover and value for leaf nodes or internal nodes
        cover = tree.get("leaf_count", 0) or tree.get("internal_count", 0)
        value = tree.get("leaf_value", 0) or tree.get("threshold", 0)
        return Node(
            cover=cover,
            is_leaf=is_leaf,
            default_left=tree.get("default_left", 0),
            feature=tree.get("split_feature"),
            value=value,
            n_children=n_children,
            left_child=left_child,
            right_child=right_child,
        )

    @staticmethod
    def from_treelite_dict(dict_all_nodes: list[dict[str, Any]], node_id: int) -> "Node":
        this_node = dict_all_nodes[node_id]
        is_leaf = "leaf_value" in this_node
        default_left = this_node.get("default_left", False)

        n_children = 0
        if "left_child" in this_node:
            left_child = Node.from_treelite_dict(dict_all_nodes, this_node["left_child"])
            n_children += 1 + left_child.n_children
        else:
            left_child = None
        if "right_child" in this_node:
            right_child = Node.from_treelite_dict(
                dict_all_nodes, this_node["right_child"]
            )
            n_children += 1 + right_child.n_children
        else:
            right_child = None

        value = this_node["leaf_value"] if is_leaf else this_node["threshold"]
        if not is_leaf:
            comp = this_node["comparison_op"]
            if comp == "<=":
                value = float(np.nextafter(value, np.inf))
            elif comp in [">", ">="]:
                left_child, right_child = right_child, left_child
                default_left = not default_left
                if comp == ">":
                    value = float(np.nextafter(value, -np.inf))
            elif comp != "<":
                raise TypeError(
                    f"Model to convert contains unsupported split type: {comp}."
                )

        return Node(
            cover=this_node.get("sum_hess", 0.0),
            is_leaf=is_leaf,
            default_left=default_left,
            feature=this_node.get("split_feature_id"),
            value=value,
            n_children=n_children,
            left_child=left_child,
            right_child=right_child,
        )

    def get_value_closest_float_downward(self) -> np.float64:
        """Get the closest exact fp value smaller than self.value"""
        return np.nextafter(np.single(self.value), np.single(-np.inf))

    def get_children(self) -> "Optional[Tuple[Node, Node]]":
        if not self.left_child or not self.right_child:
            assert self.is_leaf
        else:
            return (self.left_child, self.right_child)

    @property
    def feature(self) -> int:
        if isinstance(self.__feature, int):
            return self.__feature
        if isinstance(self.__feature, str) and self.__feature.isnumeric():
            return int(self.__feature)
        raise AttributeError(
            f"Feature names must be integers (got ({type(self.__feature)}){self.__feature})"
        )


class TreeView:
    """Helper class, treating a list of nodes as one tree"""

    def __init__(self, tree_id: int, root_node: Node) -> None:
        self.tree_id = tree_id
        self.root_node = root_node

    @property
    def is_leaf(self) -> bool:
        return self.root_node.is_leaf

    @property
    def value(self) -> float:
        if not self.is_leaf:
            raise AttributeError("Tree is not a leaf-only tree")
        if self.root_node.value is None:
            raise AttributeError("Tree is leaf-only but leaf node has no value")
        return self.root_node.value

    @property
    def cover(self) -> float:
        if not self.is_leaf:
            raise AttributeError("Tree is not a leaf-only tree")
        return self.root_node.cover

    @property
    def n_nodes(self) -> int:
        return self.root_node.n_children + 1


class TreeList(list):
    """Helper class that is able to extract all information required by the
    model builders from various objects"""

    @staticmethod
    def from_xgb_booster(
        booster, max_trees: int, feature_names_to_indices: dict[str, int]
    ) -> "TreeList":
        """
        Load a TreeList from an xgb.Booster object
        Note: We cannot type-hint the xgb.Booster without loading xgb as dependency in pyx code,
              therefore not type hint is added.
        """

        # Note: in XGBoost, it's possible to use 'int' type for features that contain
        # non-integer floating points. In such case, the training procedure and JSON
        # export from XGBoost will not treat them any differently from 'q'-type
        # (numeric) features, but the per-tree JSON text dumps used here will output
        # a split threshold rounded to the nearest integer for those 'int' features,
        # even if the booster internally has thresholds with decimal points and outputs
        # them as such in the full-model JSON dumps. Hence the need for this override
        # mechanism. If this behavior changes in XGBoost, then this conversion and
        # override can be removed.
        orig_feature_types = None
        try:
            if hasattr(booster, "feature_types"):
                feature_types = booster.feature_types
                orig_feature_types = deepcopy(feature_types)
                if feature_types:
                    for i in range(len(feature_types)):
                        if feature_types[i] == "int":
                            feature_types[i] = "float"
                    booster.feature_types = feature_types

            tl = TreeList()
            dump = booster.get_dump(dump_format="json", with_stats=True)
        finally:
            if orig_feature_types:
                booster.feature_types = orig_feature_types
        for tree_id, raw_tree in enumerate(dump):
            if max_trees > 0 and tree_id == max_trees:
                break
            raw_tree_parsed = json.loads(raw_tree)
            root_node = Node.from_xgb_dict(raw_tree_parsed, feature_names_to_indices)
            tl.append(TreeView(tree_id=tree_id, root_node=root_node))

        return tl

    @staticmethod
    def from_lightgbm_booster_dump(dump: Dict[str, Any]) -> "TreeList":
        """
        Load a TreeList from a lgbm Booster dump
        Note: We cannot type-hint the the Model without loading lightgbm as dependency in pyx code,
              therefore not type hint is added.
        """
        tl = TreeList()
        for tree_id, tree_dict in enumerate(dump["tree_info"]):
            root_node = Node.from_lightgbm_dict(tree_dict)
            tl.append(TreeView(tree_id=tree_id, root_node=root_node))

        return tl

    @staticmethod
    def from_treelite_dict(tl_json: Dict[str, Any]) -> "TreeList":
        tl = TreeList()
        for tree_id, tree_dict in enumerate(tl_json["trees"]):
            root_node = Node.from_treelite_dict(tree_dict["nodes"], 0)
            tl.append(TreeView(tree_id=tree_id, root_node=root_node))
        return tl

    def __setitem__(self):
        raise NotImplementedError(
            "Use TreeList.from_*() methods to initialize a TreeList"
        )


def get_lightgbm_params(booster):
    return booster.dump_model()


def get_xgboost_params(booster):
    return json.loads(booster.save_config())


def get_catboost_params(booster):
    with NamedTemporaryFile() as fp:
        booster.save_model(fp.name, "json")
        fp.seek(0)
        model_data = json.load(fp)
    return model_data


def get_gbt_model_from_tree_list(
    tree_list: TreeList,
    n_iterations: int,
    is_regression: bool,
    n_features: int,
    n_classes: int,
    base_score: Optional[float] = None,
):
    """Return a GBT Model from TreeList"""

    if is_regression:
        mb = gbt_reg_model_builder(n_features=n_features, n_iterations=n_iterations)
    else:
        mb = gbt_clf_model_builder(
            n_features=n_features, n_iterations=n_iterations, n_classes=n_classes
        )

    class_label = 0
    for counter, tree in enumerate(tree_list, start=1):
        # find out the number of nodes in the tree
        if is_regression:
            tree_id = mb.create_tree(tree.n_nodes)
        else:
            tree_id = mb.create_tree(n_nodes=tree.n_nodes, class_label=class_label)

        if counter % n_iterations == 0:
            class_label += 1

        if tree.is_leaf:
            mb.add_leaf(tree_id=tree_id, response=tree.value, cover=tree.cover)
            continue

        root_node = tree.root_node
        parent_id = mb.add_split(
            tree_id=tree_id,
            feature_index=root_node.feature,
            feature_value=root_node.get_value_closest_float_downward(),
            cover=root_node.cover,
            default_left=root_node.default_left,
        )

        # create queue
        node_queue: Deque[Node] = deque()
        children = root_node.get_children()
        assert children is not None
        for position, child in enumerate(children):
            child.parent_id = parent_id
            child.position = position
            node_queue.append(child)

        while node_queue:
            node = node_queue.popleft()
            assert node.parent_id != -1, "node.parent_id must not be -1"
            assert node.position != -1, "node.position must not be -1"

            if node.is_leaf:
                mb.add_leaf(
                    tree_id=tree_id,
                    response=node.value,
                    cover=node.cover,
                    parent_id=node.parent_id,
                    position=node.position,
                )
            else:
                parent_id = mb.add_split(
                    tree_id=tree_id,
                    feature_index=node.feature,
                    feature_value=node.get_value_closest_float_downward(),
                    cover=node.cover,
                    default_left=node.default_left,
                    parent_id=node.parent_id,
                    position=node.position,
                )

                children = node.get_children()
                assert children is not None
                for position, child in enumerate(children):
                    child.parent_id = parent_id
                    child.position = position
                    node_queue.append(child)

    return mb.model(base_score=base_score)


def get_gbt_model_from_lightgbm(model: Any, booster=None) -> Any:
    model_str = model.model_to_string()
    if "is_linear=1" in model_str:
        raise TypeError("Linear trees are not supported.")
    if "[boosting: dart]" in model_str:
        raise TypeError(
            "'Dart' booster is not supported. Try converting to 'treelite' first."
        )
    if "[boosting: rf]" in model_str:
        raise TypeError("Random forest boosters are not supported.")
    if ("[objective: lambdarank]" in model_str) or (
        "[objective: rank_xendcg]" in model_str
    ):
        raise TypeError("Ranking objectives are not supported.")

    if booster is None:
        booster = model.dump_model()

    n_features = booster["max_feature_idx"] + 1
    n_iterations = len(booster["tree_info"]) / booster["num_tree_per_iteration"]
    n_classes = booster["num_tree_per_iteration"]

    is_regression = False
    objective_fun = booster["objective"]
    if n_classes > 2:
        if ("ova" in objective_fun) or ("ovr" in objective_fun):
            raise TypeError(
                "Only multiclass (softmax) objective is supported for multiclass classification"
            )
    elif "binary" in objective_fun:  # nClasses == 1
        n_classes = 2
    else:
        is_regression = True

    tree_list = TreeList.from_lightgbm_booster_dump(booster)

    return get_gbt_model_from_tree_list(
        tree_list,
        n_iterations=n_iterations,
        is_regression=is_regression,
        n_features=n_features,
        n_classes=n_classes,
    )


def get_gbt_model_from_xgboost(booster: Any, xgb_config=None) -> Any:
    # Note: in the absence of any feature names, XGBoost will generate
    # tree json dumps where features are named 'f0..N'. While the JSONs
    # of the whole model will have feature indices, the per-tree JSONs
    # used here always use string names instead, hence the need for this.
    feature_names = booster.feature_names
    if feature_names:
        feature_names_to_indices = {fname: ind for ind, fname in enumerate(feature_names)}
    else:
        feature_names_to_indices = {
            f"f{ind}": ind for ind in range(booster.num_features())
        }

    if xgb_config is None:
        xgb_config = get_xgboost_params(booster)

    if xgb_config["learner"]["learner_train_param"]["booster"] != "gbtree":
        raise TypeError(
            "Only 'gbtree' booster type is supported. For DART, try converting to 'treelite' first."
        )

    n_targets = xgb_config["learner"]["learner_model_param"].get("num_target")
    if n_targets is not None and int(n_targets) > 1:
        raise TypeError("Multi-target boosters are not supported.")

    n_features = int(xgb_config["learner"]["learner_model_param"]["num_feature"])
    n_classes = int(xgb_config["learner"]["learner_model_param"]["num_class"])
    base_score = float(xgb_config["learner"]["learner_model_param"]["base_score"])

    is_regression = False
    objective_fun = xgb_config["learner"]["learner_train_param"]["objective"]

    # Note: the base score from XGBoost is in the response scale, but the predictions
    # are calculated in the link scale, so when there is a non-identity link function,
    # it needs to be converted to the link scale.
    if objective_fun in ["count:poisson", "reg:gamma", "reg:tweedie", "survival:aft"]:
        base_score = float(np.log(base_score))
    elif objective_fun == "reg:logistic":
        base_score = float(np.log(base_score / (1 - base_score)))
    elif objective_fun.startswith("rank"):
        raise TypeError("Ranking objectives are not supported.")

    if n_classes > 2:
        if objective_fun not in ["multi:softprob", "multi:softmax"]:
            raise TypeError(
                "multi:softprob and multi:softmax are only supported for multiclass classification"
            )
    elif objective_fun.startswith("binary:"):
        if objective_fun not in ["binary:logistic", "binary:logitraw"]:
            raise TypeError(
                "only binary:logistic and binary:logitraw are supported for binary classification"
            )
        n_classes = 2
        if objective_fun == "binary:logitraw":
            # daal4py always applies a sigmoid for pred_proba, wheres XGBoost
            # returns raw predictions with logitraw
            base_score = float(1 / (1 + np.exp(-base_score)))
    else:
        is_regression = True

    # max_trees=0 if best_iteration does not exist
    max_trees = getattr(booster, "best_iteration", -1) + 1
    if n_classes > 2:
        max_trees *= n_classes
    tree_list = TreeList.from_xgb_booster(booster, max_trees, feature_names_to_indices)

    if hasattr(booster, "best_iteration"):
        n_iterations = booster.best_iteration + 1
    else:
        n_iterations = len(tree_list) // (n_classes if n_classes > 2 else 1)

    return get_gbt_model_from_tree_list(
        tree_list,
        n_iterations=n_iterations,
        is_regression=is_regression,
        n_features=n_features,
        n_classes=n_classes,
        base_score=base_score,
    )


def __get_value_as_list(node):
    """Make sure the values are a list"""
    values = node["value"]
    if isinstance(values, (list, tuple)):
        return values
    else:
        return [values]


def __calc_node_weights_from_leaf_weights(weights):
    def sum_pairs(values):
        assert len(values) % 2 == 0, "Length of values must be even"
        return [values[i] + values[i + 1] for i in range(0, len(values), 2)]

    level_weights = sum_pairs(weights)
    result = [level_weights]
    while len(level_weights) > 1:
        level_weights = sum_pairs(level_weights)
        result.append(level_weights)
    return result[::-1]


def get_gbt_model_from_catboost(booster: Any) -> Any:
    if not booster.is_fitted():
        raise RuntimeError("Model should be fitted before exporting to daal4py.")

    model = CatBoostModelData(get_catboost_params(booster))

    if model.has_categorical_features:
        raise NotImplementedError(
            "Categorical features are not supported in daal4py Gradient Boosting Trees"
        )

    objective = booster.get_params().get("objective", "")
    if (
        "Rank" in objective
        or "Query" in objective
        or "Pair" in objective
        or objective in ["LambdaMart", "StochasticFilter", "GroupQuantile"]
    ):
        raise TypeError("Ranking objectives are not supported.")
    if "Multi" in objective and objective != "MultiClass":
        if model.is_classification:
            raise TypeError(
                "Only 'MultiClass' loss is supported for multi-class classification."
            )
        else:
            raise TypeError("Multi-output models are not supported.")

    if model.is_classification:
        mb = gbt_clf_model_builder(
            n_features=model.n_features,
            n_iterations=model.n_iterations,
            n_classes=model.n_classes,
        )
    else:
        mb = gbt_reg_model_builder(
            n_features=model.n_features, n_iterations=model.n_iterations
        )

    # Create splits array (all splits are placed sequentially)
    splits = []
    for feature in model.float_features:
        if feature["borders"]:
            for feature_border in feature["borders"]:
                splits.append(
                    {"feature_index": feature["feature_index"], "value": feature_border}
                )

    # Note: catboost models might have a 'bias' (intercept) which gets added
    # to all predictions. In the case of single-output models, this is a scalar,
    # but in the case of multi-output models such as multinomial logistic, it
    # is a vector. Since daal4py doesn't support vector-valued intercepts, this
    # adds the intercept to every terminal node instead, by dividing it equally
    # among all trees. Usually, catboost would anyway set them to zero, but it
    # still allows setting custom intercepts.
    cb_bias = booster.get_scale_and_bias()[1]
    add_intercept_to_each_node = isinstance(cb_bias, list)
    if add_intercept_to_each_node:
        cb_bias = np.array(cb_bias) / model.n_iterations
        if not model.is_classification:
            raise TypeError("Multi-output regression models are not supported.")

    def add_vector_bias(values: list[float]) -> list[float]:
        return list(np.array(values) + cb_bias)

    trees_explicit = []
    tree_symmetric = []

    all_trees_are_empty = True

    if model.is_symmetric_tree:
        for tree in model.oblivious_trees:
            tree_splits = tree.get("splits", [])
            cur_tree_depth = len(tree_splits) if tree_splits is not None else 0
            tree_symmetric.append((tree, cur_tree_depth))
    else:
        for tree in model.trees:
            n_nodes = 1

            # Check if node is a leaf (in case of stump)
            if "split" in tree:
                # Get number of trees and splits info via BFS
                # Create queue
                nodes_queue = []
                root_node = CatBoostNode(split=splits[tree["split"]["split_index"]])
                nodes_queue.append((tree, root_node))
                while nodes_queue:
                    cur_node_data, cur_node = nodes_queue.pop(0)
                    if "value" in cur_node_data:
                        cur_node.value = __get_value_as_list(cur_node_data)
                    else:
                        cur_node.split = splits[cur_node_data["split"]["split_index"]]
                        left_node = CatBoostNode()
                        right_node = CatBoostNode()
                        cur_node.left = left_node
                        cur_node.right = right_node
                        nodes_queue.append((cur_node_data["left"], left_node))
                        nodes_queue.append((cur_node_data["right"], right_node))
                        n_nodes += 2
                        all_trees_are_empty = False
            else:
                root_node = CatBoostNode()
                if model.is_classification and model.n_classes > 2:
                    root_node.value = [value * model.scale for value in tree["value"]]
                    if add_intercept_to_each_node:
                        root_node.value = add_vector_bias(root_node.value)
                else:
                    root_node.value = [tree["value"] * model.scale]
            trees_explicit.append((root_node, n_nodes))

    tree_id = []
    class_label = 0
    count = 0

    # Only 1 tree for each iteration in case of regression or binary classification
    if not model.is_classification or model.n_classes == 2:
        n_tree_each_iter = 1
    else:
        n_tree_each_iter = model.n_classes

    shap_ready = False

    # Create id for trees (for the right order in model builder)
    for i in range(model.n_iterations):
        for _ in range(n_tree_each_iter):
            if model.is_symmetric_tree:
                if not len(tree_symmetric):
                    n_nodes = 1
                else:
                    n_nodes = 2 ** (tree_symmetric[i][1] + 1) - 1
            else:
                if not len(trees_explicit):
                    n_nodes = 1
                else:
                    n_nodes = trees_explicit[i][1]

            if model.is_classification and model.n_classes > 2:
                tree_id.append(mb.create_tree(n_nodes, class_label))
                count += 1
                if count == model.n_iterations:
                    class_label += 1
                    count = 0

            elif model.is_classification:
                tree_id.append(mb.create_tree(n_nodes, 0))
            else:
                tree_id.append(mb.create_tree(n_nodes))

    if model.is_symmetric_tree:
        shap_ready = True  # this code branch provides all info for SHAP values
        for class_label in range(n_tree_each_iter):
            for i in range(model.n_iterations):
                cur_tree_info = tree_symmetric[i][0]
                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                cur_tree_leaf_val = cur_tree_info["leaf_values"]
                cur_tree_leaf_weights = cur_tree_info["leaf_weights"]
                cur_tree_depth = tree_symmetric[i][1]
                if cur_tree_depth == 0:
                    mb.add_leaf(
                        tree_id=cur_tree_id,
                        response=cur_tree_leaf_val[class_label] * model.scale
                        + (cb_bias[class_label] if add_intercept_to_each_node else 0),
                        cover=cur_tree_leaf_weights[0],
                    )
                else:
                    # One split used for the whole level
                    cur_level_split = splits[
                        cur_tree_info["splits"][cur_tree_depth - 1]["split_index"]
                    ]
                    cur_tree_weights_per_level = __calc_node_weights_from_leaf_weights(
                        cur_tree_leaf_weights
                    )
                    root_weight = cur_tree_weights_per_level[0][0]

                    root_id = mb.add_split(
                        tree_id=cur_tree_id,
                        feature_index=cur_level_split["feature_index"],
                        feature_value=cur_level_split["value"],
                        default_left=model.default_left,
                        cover=root_weight,
                    )
                    prev_level_nodes = [root_id]

                    # Iterate over levels, splits in json are reversed (root split is the last)
                    for cur_level in range(cur_tree_depth - 2, -1, -1):
                        cur_level_nodes = []
                        next_level_weights = cur_tree_weights_per_level[cur_level + 1]
                        cur_level_node_index = 0
                        for cur_parent in prev_level_nodes:
                            cur_level_split = splits[
                                cur_tree_info["splits"][cur_level]["split_index"]
                            ]
                            cover_nodes = next_level_weights[cur_level_node_index]
                            if cover_nodes == 0:
                                shap_ready = False
                            cur_left_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=0,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=model.default_left,
                                cover=cover_nodes,
                            )
                            # cur_level_node_index += 1
                            cur_right_node = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_parent,
                                position=1,
                                feature_index=cur_level_split["feature_index"],
                                feature_value=cur_level_split["value"],
                                default_left=model.default_left,
                                cover=cover_nodes,
                            )
                            # cur_level_node_index += 1
                            cur_level_nodes.append(cur_left_node)
                            cur_level_nodes.append(cur_right_node)
                        prev_level_nodes = cur_level_nodes

                    # Different storing format for leaves
                    if not model.is_classification or model.n_classes == 2:
                        for last_level_node_num in range(len(prev_level_nodes)):
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num]
                                * model.scale,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=0,
                                cover=cur_tree_leaf_weights[2 * last_level_node_num],
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[2 * last_level_node_num + 1]
                                * model.scale,
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
                                cover=cur_tree_leaf_weights[2 * last_level_node_num + 1],
                            )
                    else:
                        shap_ready = False
                        for last_level_node_num in range(len(prev_level_nodes)):
                            left_index = (
                                2 * last_level_node_num * n_tree_each_iter + class_label
                            )
                            right_index = (
                                2 * last_level_node_num + 1
                            ) * n_tree_each_iter + class_label
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[left_index] * model.scale
                                + (
                                    cb_bias[class_label]
                                    if add_intercept_to_each_node
                                    else 0
                                ),
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=0,
                                cover=0.0,
                            )
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=cur_tree_leaf_val[right_index] * model.scale
                                + (
                                    cb_bias[class_label]
                                    if add_intercept_to_each_node
                                    else 0
                                ),
                                parent_id=prev_level_nodes[last_level_node_num],
                                position=1,
                                cover=0.0,
                            )
    else:
        shap_ready = False
        scale = booster.get_scale_and_bias()[0]
        for class_label in range(n_tree_each_iter):
            for i in range(model.n_iterations):
                root_node = trees_explicit[i][0]

                cur_tree_id = tree_id[i * n_tree_each_iter + class_label]
                # Traverse tree via BFS and build tree with modelbuilder
                if root_node.value is None:
                    root_id = mb.add_split(
                        tree_id=cur_tree_id,
                        feature_index=root_node.split["feature_index"],
                        feature_value=root_node.split["value"],
                        default_left=model.default_left,
                        cover=0.0,
                    )
                    nodes_queue = [(root_node, root_id)]
                    while nodes_queue:
                        cur_node, cur_node_id = nodes_queue.pop(0)
                        left_node = cur_node.left
                        # Check if node is a leaf
                        if left_node.value is None:
                            left_node_id = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_node_id,
                                position=0,
                                feature_index=left_node.split["feature_index"],
                                feature_value=left_node.split["value"],
                                default_left=model.default_left,
                                cover=0.0,
                            )
                            nodes_queue.append((left_node, left_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=scale * left_node.value[class_label]
                                + (
                                    cb_bias[class_label]
                                    if add_intercept_to_each_node
                                    else 0
                                ),
                                parent_id=cur_node_id,
                                position=0,
                                cover=0.0,
                            )
                        right_node = cur_node.right
                        # Check if node is a leaf
                        if right_node.value is None:
                            right_node_id = mb.add_split(
                                tree_id=cur_tree_id,
                                parent_id=cur_node_id,
                                position=1,
                                feature_index=right_node.split["feature_index"],
                                feature_value=right_node.split["value"],
                                default_left=model.default_left,
                                cover=0.0,
                            )
                            nodes_queue.append((right_node, right_node_id))
                        else:
                            mb.add_leaf(
                                tree_id=cur_tree_id,
                                response=scale * cur_node.right.value[class_label]
                                + (
                                    cb_bias[class_label]
                                    if add_intercept_to_each_node
                                    else 0
                                ),
                                parent_id=cur_node_id,
                                position=1,
                                cover=0.0,
                            )

                else:
                    # Tree has only one node
                    # Note: the root node already has scale and bias added to it,
                    # so no need to add them again here like it is done for the leafs.
                    mb.add_leaf(
                        tree_id=cur_tree_id,
                        response=root_node.value[class_label],
                        cover=0.0,
                    )

    if all_trees_are_empty and not model.is_symmetric_tree:
        shap_ready = True

    intercept = 0.0
    if not add_intercept_to_each_node:
        intercept = booster.get_scale_and_bias()[1]
    return mb.model(base_score=intercept), shap_ready


def get_gbt_model_from_treelite(
    tl_model: "treelite.model.Model",
) -> tuple[Any, int, int, bool]:
    model_json = json.loads(tl_model.dump_as_json())
    task_type = model_json["task_type"]
    if task_type not in ["kBinaryClf", "kRegressor", "kMultiClf", "kIsolationForest"]:
        raise TypeError(f"Model to convert is of unsupported type: {task_type}")
    if model_json["num_target"] > 1:
        raise TypeError("Multi-target models are not supported.")
    if model_json["postprocessor"] == "multiclass_ova":
        raise TypeError(
            "Multi-class classification models that use One-Vs-All are not supported."
        )
    for tree in model_json["trees"]:
        if tree["has_categorical_split"]:
            raise TypeError("Models with categorical features are not supported.")
    num_trees = tl_model.num_tree
    if not num_trees:
        raise TypeError("Model to convert contains no trees.")

    # Note: the daal4py module always adds up the scores, but some models
    # might average them instead. In such case, this turns the trees into
    # additive ones by dividing the predictions by the number of nodes beforehand.
    if model_json["average_tree_output"]:
        divide_treelite_leaf_values_by_const(model_json, num_trees)

    base_score = model_json["base_scores"]
    num_class = model_json["num_class"][0]
    num_feature = model_json["num_feature"]

    if task_type == "kBinaryClf":
        num_class = 2
        if base_score:
            base_score = list(1 / (1 + np.exp(-np.array(base_score))))

    if num_class > 2:
        shap_ready = False
    else:
        shap_ready = True
        for tree in model_json["trees"]:
            if not tree["nodes"][0].get("sum_hess", False):
                shap_ready = False
                break

    # In the case of random forests for classification, it might work
    # by averaging predictions without any link function, whereas
    # daal4py assumes a logit link. In such case, it's not possible to
    # convert them to daal4py's logic, but the model can still be used
    # as a regressor that always outputs something between 0 and 1.
    is_regression = "Clf" not in task_type
    if not is_regression and model_json["postprocessor"] == "identity_multiclass":
        is_regression = True
        warnings.warn(
            "Attempting to convert classification model which is not"
            " based on gradient boosting. Will output a regression"
            " model instead."
        )

    looks_like_random_forest = (
        model_json["postprocessor"] == "identity_multiclass"
        and len(model_json["base_scores"]) > 1
        and task_type == "kMultiClf"
    )
    if looks_like_random_forest:
        if num_class > 2 or len(base_score) > 2:
            raise TypeError("Multi-class random forests are not supported.")
        if len(model_json["num_class"]) > 1:
            raise TypeError("Multi-output random forests are not supported.")
        if len(base_score) == 2 and base_score[0]:
            raise TypeError("Random forests with base scores are not supported.")

    # In the case of binary random forests, it will always have leaf values
    # for 2 classes, which is redundant as they sum to 1. daal4py requires
    # only values for the positive class, so they need to be converted.
    if looks_like_random_forest:
        leave_only_last_treelite_leaf_value(model_json)
        base_score = base_score[-1]

    # In the case of multi-class classification models, if converted
    # from xgboost, the order of the trees will be the same - i.e.
    # sequences of one tree of each class, followed by another such
    # sequence. But treelite could in theory also support building
    # models where the trees are in a different order, in which case
    # they will need to be reordered to match xgboost, since that's
    # how daal4py handles them. And if there is an uneven number of
    # trees per class, then will need to make up extra trees with
    # zeros to accommodate it.
    if task_type == "kMultiClf" and not looks_like_random_forest:
        num_trees = len(model_json["trees"])
        if (num_trees % num_class) != 0:
            shap_ready = False
            class_ids, num_trees_per_class = np.unique(
                model_json["class_id"], return_counts=True
            )
            max_tree_per_class = num_trees_per_class.max()
            num_tree_add_per_class = max_tree_per_class - num_trees_per_class
            for class_ind in range(num_class):
                for tree in range(num_tree_add_per_class[class_ind]):
                    add_empty_tree_to_treelite_json(model_json, class_ind)

        tree_class_orders = model_json["class_id"]
        sequential_ids = np.arange(num_class)
        num_trees = len(model_json["trees"])
        assert (num_trees % num_class) == 0
        if not np.array_equal(
            tree_class_orders, np.tile(sequential_ids, int(num_trees / num_class))
        ):
            argsorted_class_indices = np.argsort(tree_class_orders)
            per_class_indices = np.split(argsorted_class_indices, num_class)
            correct_order = np.vstack(per_class_indices).reshape(-1, order="F")
            model_json["trees"] = [model_json["trees"][ix] for ix in correct_order]
            model_json["class_id"] = [model_json["class_id"][ix] for ix in correct_order]

    # In the case of multi-class classification with base scores,
    # since daal4py only supports scalar intercepts, this follows the
    # same strategy as in catboost of dividing the intercepts equally
    # among the number of trees
    if task_type == "kMultiClf" and not looks_like_random_forest:
        add_intercept_to_treelite_leafs(model_json, base_score)
        base_score = None

    if isinstance(base_score, list):
        if len(base_score) == 1:
            base_score = base_score[0]
        else:
            raise TypeError("Model to convert is malformed.")

    tree_list = TreeList.from_treelite_dict(model_json)
    return (
        get_gbt_model_from_tree_list(
            tree_list,
            n_iterations=num_trees
            / (
                num_class
                if task_type == "kMultiClf" and not looks_like_random_forest
                else 1
            ),
            is_regression=is_regression,
            n_features=num_feature,
            n_classes=num_class,
            base_score=base_score,
        ),
        num_class,
        num_feature,
        shap_ready,
    )


def divide_treelite_leaf_values_by_const(
    tl_json: dict[str, Any], divisor: "int | float"
) -> None:
    for tree in tl_json["trees"]:
        for node in tree["nodes"]:
            if "leaf_value" in node:
                if isinstance(node["leaf_value"], (list, tuple)):
                    node["leaf_value"] = list(np.array(node["leaf_value"]) / divisor)
                else:
                    node["leaf_value"] /= divisor


def leave_only_last_treelite_leaf_value(tl_json: dict[str, Any]) -> None:
    for tree in tl_json["trees"]:
        for node in tree["nodes"]:
            if "leaf_value" in node:
                assert len(node["leaf_value"]) == 2
                node["leaf_value"] = node["leaf_value"][-1]


def add_intercept_to_treelite_leafs(
    tl_json: dict[str, Any], base_score: list[float]
) -> None:
    num_trees_per_class = len(tl_json["trees"]) / tl_json["num_class"][0]
    for tree_index, tree in enumerate(tl_json["trees"]):
        leaf_add = base_score[tl_json["class_id"][tree_index]] / num_trees_per_class
        for node in tree["nodes"]:
            if "leaf_value" in node:
                node["leaf_value"] += leaf_add


def add_empty_tree_to_treelite_json(tl_json: dict[str, Any], class_add: int) -> None:
    tl_json["class_id"].append(class_add)
    tl_json["trees"].append(
        {
            "num_nodes": 1,
            "has_categorical_split": False,
            "nodes": [
                {
                    "node_id": 0,
                    "leaf_value": 0.0,
                    "data_count": 0,
                    "sum_hess": 0.0,
                },
            ],
        }
    )

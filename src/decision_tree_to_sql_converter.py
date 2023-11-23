"""
Decision Tree to SQL Converter

This file contains functions to convert a trained Decision Tree Classifier to a JSON representation and then
to an SQL query. This can be useful for deploying a simple decision tree model in a database or an environment
where Python execution is not feasible.

Author: Aliaksandra Strunova
"""

import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification


def _tree_as_dict(tree, node_index: int) -> dict:
    """
    Recursively convert a decision tree node to a dictionary.

    Args:
        tree: The decision tree classifier.
        node_index (int): The index of the current node in the tree.

    Returns:
        dict: A dictionary representation of the node.
    """
    if tree.children_left[node_index] == tree.children_right[node_index] == -1:
        return {"class": int(tree.value[node_index].argmax())}
    return {
        "feature_index": int(tree.feature[node_index]),
        "threshold": float(round(tree.threshold[node_index], 4)),
        "left": _tree_as_dict(tree, tree.children_left[node_index]),
        "right": _tree_as_dict(tree, tree.children_right[node_index])
    }


def convert_tree_to_json(tree: DecisionTreeClassifier) -> str:
    """
    Convert a DecisionTreeClassifier to a JSON-formatted string.

    Args:
        tree (DecisionTreeClassifier): The trained decision tree classifier.

    Returns:
        str: JSON string representation of the decision tree.
    """
    tree_dict = _tree_as_dict(tree.tree_, node_index=0)
    tree_as_json = json.dumps(tree_dict, indent=4)
    print(tree_as_json)

    return tree_as_json


def process_node(node) -> str:
    """
    Convert a node of the tree (in dictionary form) to a part of an SQL query.

    Args:
        node (dict): A node of the decision tree.

    Returns:
        str: SQL query fragment representing the decision logic of the node.
    """
    if 'feature_index' in node:
        left_sql = process_node(node['left'])
        right_sql = process_node(node['right'])
        sql = (
            f"CASE WHEN feature_{node['feature_index']} > {node['threshold']} "
            f"THEN {left_sql} ELSE {right_sql} END"
        )
        return sql
    else:
        return str(node['class'])


def generate_sql_query(tree_as_json: str, features: list = []) -> str:
    """
    Generate an SQL query for a decision tree given its JSON representation.

    Args:
        tree_as_json (str): The JSON string representation of the decision tree.
        features (list): List of feature indices to be included in the query.

    Returns:
        str: The complete SQL query to execute the decision tree logic.
    """
    json_as_dict = json.loads(tree_as_json)

    sql_query = "SELECT "
    for feature in features:
        if json_as_dict['feature_index'] == feature:
            sql = process_node(json_as_dict)
            sql_query += f"{sql}"

    sql_query += f" as CLASS_LABEL"

    return sql_query


# Example usage
# Generate a synthetic classification dataset
X, y = make_classification(n_samples=100,
                           n_features=5,
                           n_informative=2,
                           n_classes=2,
                           random_state=42)

# Train decision tree
model = DecisionTreeClassifier(max_depth=3)
model.fit(X, y)

# Convert the tree to JSON and then to an SQL query
_json = convert_tree_to_json(model)
_sql = generate_sql_query(_json, [3, 2])
print(_sql)

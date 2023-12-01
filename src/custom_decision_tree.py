"""
This module implements a simple Decision Tree Regressor. It can build a decision tree
from a training dataset, predict values, and convert the tree to a JSON format.

Key Components:
- DecisionTreeRegressor: Main class to build and use a decision tree for regression.
- Node: Represents a node in the decision tree.
- CustomEncoder: Custom JSON encoder for serializing Node instances.

The DecisionTreeRegressor class includes methods to fit the model to the training data,
predict values, and serialize the tree to JSON. It utilizes mean squared error for determining
the best split at each node.

Example:
- Train a decision tree regressor on a dataset, predict values, and convert the tree to JSON.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import json


@dataclass
class DecisionTreeRegressor:
    """Decision tree regressor."""
    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters:
            X (np.ndarray): Input features for training.
            y (np.ndarray): Target values for training.

        Returns:
            DecisionTreeRegressor: The instance itself.
        """
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def _mse(self, y: np.ndarray) -> float:
        """
        Compute the mean squared error criterion for a given set of target values.

        Parameters:
            y (np.ndarray): Target values.

        Returns:
            float: Computed mean squared error.
        """
        y_mean = np.mean(y)
        if len(y) == 0:
            return 0
        return np.sum(np.square(y - y_mean)) / len(y)

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """
        Compute the weighted mean squared error criterion for two sets of target values.

        Parameters:
            y_left (np.ndarray): Target values for the left split.
            y_right (np.ndarray): Target values for the right split.

        Returns:
            float: Computed weighted mean squared error.
        """
        y_left_len = len(y_left)
        y_right_len = len(y_right)
        if (y_left_len + y_right_len) == 0:
            return 0
        return (self._mse(y_left) * y_left_len + self._mse(y_right) * y_right_len) / (y_left_len + y_right_len)

    def _split(self, X: np.ndarray, y: np.ndarray, feature: int):
        """
        Find the best split for a node using a single feature.

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            feature (int): Index of the feature used for splitting.

        Returns:
            A tuple containing the best threshold for the split and the score of that split.
        """
        feature_data = X[:, feature]
        unique_values = np.unique(feature_data)
        best_threshold, best_score = 0, None
        for current_threshold in unique_values:
            mask_left = feature_data <= current_threshold
            mask_right = feature_data > current_threshold

            if len(y[mask_left]) > 0 and len(y[mask_right]) > 0:
                current_score = self._weighted_mse(y[mask_left], y[mask_right])
                if best_score is None or current_score < best_score:
                    best_score = current_score
                    best_threshold = current_threshold
        return best_threshold, best_score

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """
        Find the best split for a node across all features.

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.

        Returns:
            A tuple of the best feature index for the split, the best threshold, and the score of that split.
        """
        best_feature, best_threshold, best_score = None, None, None
        for index in range(X.shape[1]):
            current_threshold, current_score = self._split(X, y, index)
            if best_score is None or current_score < best_score:
                best_score = current_score
                best_threshold = current_threshold
                best_feature = index
        return best_feature, best_threshold

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively split a node and return the resulting left and right child nodes.

        Parameters:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
            depth (int): Current depth of the node in the tree.

        Returns:
            Node: The resulting node after splitting.
        """
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            return Node(n_samples=len(X), value=int(np.round(np.mean(y))), mse=float(np.round(self._mse(y), 2)))

        index, threshold = self._best_split(X, y)
        feature_data = X[:, index]
        mask_left = feature_data <= threshold
        mask_right = feature_data > threshold
        node = Node(feature=int(index),
                    threshold=int(threshold),
                    n_samples=len(X),
                    value=int(np.round(np.mean(y))),
                    mse=float(self._mse(y)),
                    left=self._split_node(X[mask_left], y[mask_left], depth=depth + 1),
                    right=self._split_node(X[mask_right], y[mask_right], depth=depth + 1))
        return node

    def _as_json(self, node: Node) -> dict:
        """Return the decision tree as a dictionary. Execute recursively."""
        if node.left is None and node.right is None:  # Check if it's a leaf node
            return {
                "n_samples": node.n_samples,
                "value": node.value,
                "mse": node.mse
            }
        else:
            return {
                "feature": node.feature,
                "threshold": node.threshold,
                "n_samples": node.n_samples,
                # "value": node.value,
                "mse": np.round(node.mse, 2),
                "left": self._as_json(node.left) if node.left else None,
                "right": self._as_json(node.right) if node.right else None
            }

    def as_json(self):
        """
        Convert the trained decision tree into a JSON-formatted string.

        Returns:
            str: A JSON string representation of the decision tree.
        """
        return json.dumps(self._as_json(self.tree_), cls=CustomEncoder, indent=4)

    def _get_value(self, node: Node, features: np.ndarray):
        """
        Recursively traverse the tree to find the value for a given set of features.

        Parameters:
            node (Node): The current node in the tree.
            features (np.ndarray): The input features for which the value is to be predicted.

        Returns:
            int: The predicted value at the leaf node.
        """
        if node.left is None and node.right is None:
            return node.value
        elif features[node.feature] <= node.threshold:
            return self._get_value(node.left, features)
        else:
            return self._get_value(node.right, features)

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """
        Predict the target value for a single sample.

        Parameters:
            features (np.ndarray): The input features for a single sample.

        Returns:
            int: The predicted target value.
        """
        return self._get_value(self.tree_, features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression targets for multiple samples.

        Parameters:
            X (np.ndarray): The input features for multiple samples.

        Returns:
            np.ndarray: An array of predicted target values.
        """
        y = np.empty(len(X))
        for index, sample in enumerate(X):
            y[index] = self._predict_one_sample(sample)
        return y


@dataclass
class Node:
    """Decision tree node."""
    feature: int = None
    threshold: int = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None


class CustomEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for serializing Node instances.

    Overrides the default method to enable serialization of custom dataclass objects.
    """

    def default(self, obj):
        if isinstance(obj, Node):
            return obj.__dict__  # Convert Node instance to a dictionary
        return super().default(obj)

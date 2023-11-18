"""
Gradient Boosting Regressor Implementation in Python

This file contains an implementation of a Gradient Boosting Regressor from scratch using numpy and sklearn.
The regressor uses decision trees as the base learners. It supports subsampling and can be configured for different
loss functions, learning rates, and tree parameters.

Author: Aliaksandra Strunova
"""

from typing import Tuple

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample


class GradientBoostingRegressor:
    def __init__(
            self,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=2,
            loss="mse",
            verbose=False,
            subsample_size=0.5,
            replace=False
    ):
        """
        Initialize the GradientBoostingRegressor.

        Args:
            n_estimators (int): The number of boosting stages to be run.
            learning_rate (float): Rate at which the model adapts to the problem.
            max_depth (int): Maximum depth of the individual regression estimators.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            loss (str or callable): The loss function to be used. 'mse' for mean squared error.
            verbose (bool): Enable verbose output.
            subsample_size (float): The fraction of samples to be used for fitting the individual base learners.
            replace (bool): Whether to sample with replacement.

        Returns:
            None
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.verbose = verbose
        self.subsample_size = subsample_size
        self.replace = replace

    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Calculate mean squared error and its gradient.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.

        Returns:
            Tuple[float, np.ndarray]: A tuple containing the MSE and its gradient.
        """
        loss = np.sum(np.square(y_pred - y_true)) / len(y_pred)
        grad = y_pred - y_true

        return loss, grad

    def _subsample(self, X, y):
        """
        Resample X and Y based on the subsample size.

        Args:
            X (np.ndarray): Features data.
            y (np.ndarray): Target data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Subsampled feature and target data.
        """
        combined_data = np.column_stack((X, y))
        sub_size = int(np.floor(len(combined_data) * self.subsample_size))
        subsample = resample(combined_data, replace=self.replace, n_samples=sub_size)
        sub_X = subsample[:, :-1]
        sub_y = subsample[:, -1]
        return sub_X, sub_y

    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray, loss_f="mse") -> Tuple[float, np.array]:
        """
        Calculate loss and gradient based on the specified loss function.

        Args:
            y_true (np.ndarray): True target values.
            y_pred (np.ndarray): Predicted target values.
            loss_f (str or callable): Loss function to use.

        Returns:
            Tuple[float, np.array]: A tuple containing the loss and its gradient.
        """
        if callable(loss_f):
            loss, grad = loss_f(y_true, y_pred)
        elif loss_f == 'mse':
            loss, grad = self._mse(y_true, y_pred)
        else:
            raise ValueError(f'{self.loss} loss function is not supported')

        return loss, grad

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X (np.ndarray): Feature data.
            y (np.ndarray): Target data.

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.trees_ = []
        self.base_pred_ = float(np.mean(y))
        prediction = np.full(len(X), self.base_pred_)

        for tree in range(self.n_estimators):
            loss, grad = self._loss(y, prediction, self.loss)
            anti_grad = grad * (-1)

            if self.subsample_size < 1:
                sub_X, sub_antigrad = self._subsample(X, anti_grad)
            else:
                sub_X, sub_antigrad = X, anti_grad

            tree_model = DecisionTreeRegressor(max_depth=self.max_depth,
                                               min_samples_split=self.min_samples_split)
            tree_model.fit(sub_X, sub_antigrad)
            current_prediction = tree_model.predict(X)
            prediction += self.learning_rate * current_prediction
            if self.verbose:
                print(f'tree # {tree}, prediction: {prediction}')
            self.trees_.append(tree_model)

        return self

    def predict(self, X):
        """
        Predict the target for new data.

        Args:
            X (np.ndarray): Feature data for which to make predictions.

        Returns:
            np.ndarray: Predicted target values.
        """
        predictions = np.full(len(X), self.base_pred_)
        for tree in self.trees_:
            current_prediction = tree.predict(X)
            predictions += self.learning_rate * current_prediction
            if self.verbose:
                print(f'current: {current_prediction}, new: {predictions}')

        return predictions

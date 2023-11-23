"""
This script implements matrix factorization techniques to generate item embeddings
from a given User-Item matrix. It includes classes for building the User-Item matrix,
normalizing it, and functions for obtaining item embeddings using both SVD and ALS methods.
The generated embeddings can be used in recommendation systems and similar applications.
"""

import pickle
from typing import Dict

import implicit
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


class UserItemMatrix:
    def __init__(self, sales_data: pd.DataFrame):
        """Class initialization. You can make necessary
        calculations here.

        Args:
            sales_data (pd.DataFrame): Sales dataset.

        Example:
            sales_data (pd.DataFrame):
                user_id  item_id  qty   price

        """
        self._sales_data = sales_data.copy()

        # Mapping user and item IDs to matrix indices
        self._user_count = self._sales_data['user_id'].nunique()
        self._item_count = self._sales_data['item_id'].nunique()
        self._user_map = {user: index for index, user in
                          enumerate(sorted(self._sales_data['user_id'].unique().tolist()))}
        self._item_map = {item: index for index, item in
                          enumerate(sorted(self._sales_data['item_id'].unique().tolist()))}

        # Building the CSR matrix
        self._matrix = csr_matrix((self._sales_data['qty'].values,
                                   (self._sales_data['user_id'].map(self._user_map).values,
                                    self._sales_data['item_id'].map(self._item_map).values)),
                                  shape=(self._user_count, self._item_count)
                                  )

    # Accessor methods for the class properties
    @property
    def user_count(self) -> int:
        """
        Returns:
            int: the number of users in sales_data.
        """
        return self._user_count

    @property
    def item_count(self) -> int:
        """
        Returns:
            int: the number of items in sales_data.
        """
        return self._item_count

    @property
    def user_map(self) -> Dict[int, int]:
        """Creates a mapping from user_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            user_map (Dict[int, int]):
                {1: 0, 2: 1, 4: 2, 5: 3}

        Returns:
            Dict[int, int]: User map
        """
        return self._user_map

    @property
    def item_map(self) -> Dict[int, int]:
        """Creates a mapping from item_id to matrix rows indexes.

        Example:
            sales_data (pd.DataFrame):

                user_id  item_id  qty   price
            0        1      118    1   626.66
            1        1      285    1  1016.57
            2        2     1229    3   518.99
            3        4     1688    2   940.84
            4        5     2068    1   571.36

            item_map (Dict[int, int]):
                {118: 0, 285: 1, 1229: 2, 1688: 3, 2068: 4}

        Returns:
            Dict[int, int]: Item map
        """
        return self._item_map

    @property
    def csr_matrix(self) -> csr_matrix:
        """User items matrix in form of CSR matrix.

        User row_ind, col_ind as
        rows and cols indices(mapped from user/item map).

        Returns:
            csr_matrix: CSR matrix
        """
        return self._matrix


class Normalization:
    @staticmethod
    def by_column(matrix: csr_matrix) -> csr_matrix:
        """Normalization by column

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        col_sums = matrix.sum(axis=0)
        norm_matrix = matrix.multiply(1 / col_sums).tocsr()
        return norm_matrix

    @staticmethod
    def by_row(matrix: csr_matrix) -> csr_matrix:
        """Normalization by row

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        row_sums = matrix.sum(axis=1)
        norm_matrix = matrix.multiply(1 / row_sums).tocsr()
        return norm_matrix

    @staticmethod
    def tf_idf(matrix: csr_matrix) -> csr_matrix:
        """Normalization using tf-idf

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        tf = Normalization.by_row(matrix)
        idf = np.log(matrix.shape[0] / (matrix > 0).sum(axis=0))
        norm_matrix = tf.multiply(idf).tocsr()
        return norm_matrix

    @staticmethod
    def bm_25(
            matrix: csr_matrix, k1: float = 2.0, b: float = 0.75
    ) -> csr_matrix:
        """Normalization based on BM-25

        Args:
            matrix (csr_matrix): User-Item matrix of size (N, M)

        Returns:
            csr_matrix: Normalized matrix of size (N, M)
        """
        n_users = matrix.shape[0]
        item_length = matrix.sum(axis=1)
        avg_item_length = np.mean(item_length)

        tf = Normalization.by_row(matrix)
        idf = np.log(n_users / (matrix > 0).sum(axis=0))

        delta = k1 * (1 - b + b * item_length / avg_item_length)
        tf_prime = tf.multiply(1 / delta).power(-1)
        tf_prime.data += 1
        tf_prime = tf_prime.power(-1) * (k1 + 1)

        norm_matrix = tf_prime.multiply(idf).tocsr()
        return norm_matrix


def items_embeddings_svd(ui_matrix: csr_matrix, dim: int) -> np.ndarray:
    """Build items embedding using Truncated SVD.

    Args:
        ui_matrix (pd.DataFrame): User-Item matrix of size (N, M)
        dim (int): Dimension of embedding vectors

    Returns:
        np.ndarray: Items embeddings matrix of size (M, dim)
    """
    # Initialize Truncated SVD with specified number of components (dim)
    svd = TruncatedSVD(n_components=dim)

    # Perform matrix factorization on the transpose of the User-Item matrix
    # Transpose is used to get item embeddings (M, dim) instead of user embeddings (N, dim)
    item_embeddings = svd.fit_transform(ui_matrix.T)

    return item_embeddings


def items_embeddings_als(ui_matrix: csr_matrix, dim: int, **kwargs) -> np.ndarray:
    """Build items embedding using ALS.

    Args:
        ui_matrix (pd.DataFrame): User-Item matrix of size (N, M)
        dim (int): Dimension of embedding vectors

    Returns:
        np.ndarray: Items embeddings matrix of size (M, dim)
    """
    # Initialize the ALS model
    model = implicit.als.AlternatingLeastSquares(factors=dim,
                                                 regularization=kwargs.get('regularization', 0.05),
                                                 iterations=kwargs.get('iterations', 4),
                                                 random_state= kwargs.get('seed', 0)
                                                 )

    # Train the model on the item-user matrix
    model.fit(ui_matrix, show_progress=False)

    # Get the item embeddings
    # The item_factors attribute holds the item embeddings
    item_embeddings = model.item_factors

    return item_embeddings


if __name__ == "__main__":
    # Load data and generate item embeddings using SVD and ALS
    df = pd.read_csv('../data/custom_user_item_embedding.csv')
    user_item_crs_matrix = UserItemMatrix(df).csr_matrix
    crs_matrix_normalized = Normalization().bm_25(user_item_crs_matrix)
    item_embeddings_svd = items_embeddings_svd(crs_matrix_normalized, 2)
    items_embeddings_als = items_embeddings_als(crs_matrix_normalized, 2)

    # Save the embeddings to files
    with open('../predictions/custom_items_embeddings_svd.pkl', 'wb') as file:
        pickle.dump(items_embeddings_svd, file)
    with open('../predictions/custom_item_embeddings_als.pkl', 'wb') as file:
        pickle.dump(items_embeddings_als, file)
"""
This script processes item embeddings for recommendation systems, featuring:

1. Similarity Calculation:
   Computes pairwise cosine similarities between items based on embeddings.
   This is useful for assessing item relationships in embedding space.

2. K-Nearest Neighbors (KNN):
   Identifies 'k' closest items for each item using similarity scores.
   Crucial for finding similar items in recommendation algorithms.

3. Weighted Average Price Calculation:
   Calculates a new weighted average price for each item.
   Prices of nearest neighbors and their similarity scores are used as weights.

4. Transform Function:
   Combines the above to map each item to its new weighted average price.
   Useful for price recommendations based on item similarities.

The script uses numpy for numerical operations and scipy for cosine calculations,
ensuring efficiency. Example usage with sample data is provided at the end.
"""

from itertools import combinations
from typing import Dict, Tuple, List

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


class SimilarItems:
    @staticmethod
    def similarity(embeddings: Dict[int, np.ndarray]) -> Dict[Tuple[int, int], float]:
        """Calculate pairwise similarities between each item
        in embedding.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.

        Returns:
            Tuple[List[str], Dict[Tuple[int, int], float]]:
            List of all items + Pairwise similarities dict
            Keys are in form of (i, j) - combinations pairs of item_ids
            with i < j.
            Round each value to 8 decimal places.
        """
        ids_list = list(embeddings.keys())
        pair_sims = {}

        for id_1, id_2 in combinations(ids_list, 2):
            # Convert cosine distance to similarity
            similarity = 1 - cosine(embeddings[id_1], embeddings[id_2])
            pair_sims[(id_1, id_2)] = round(similarity, 8)

        return pair_sims

    @staticmethod
    def knn(
            sim: Dict[Tuple[int, int], float], top: int
    ) -> Dict[int, List[Tuple[int, float]]]:
        """Return closest neighbors for each item.

        Args:
            sim (Dict[Tuple[int, int], float]): <similarity> method output.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, List[Tuple[int, float]]]: Dict with top closest neighbors
            for each item.
        """
        knn_dict = {}

        for (id_1, id_2), similarity in sim.items():
            knn_dict.setdefault(id_1, []).append((id_2, sim[(id_1, id_2)]))
            knn_dict.setdefault(id_2, []).append((id_1, sim[(id_1, id_2)]))

        for item_id in knn_dict:
            knn_dict[item_id] = sorted(knn_dict[item_id], key=lambda item: item[1], reverse=True)[:top]

        return knn_dict

    @staticmethod
    def knn_price(
            knn_dict: Dict[int, List[Tuple[int, float]]],
            prices: Dict[int, float],
    ) -> Dict[int, float]:
        """Calculate weighted average prices for each item.
        Weights should be positive numbers in [0, 2] interval.

        Args:
            knn_dict (Dict[int, List[Tuple[int, float]]]): <knn> method output.
            prices (Dict[int, float]): Price dict for each item.

        Returns:
            Dict[int, float]: New prices dict, rounded to 2 decimal places.
        """
        knn_price_dict = {}
        for item_id, weights in knn_dict.items():
            total_weight = sum([similarity + 1 for _, similarity in weights])
            weighted_price = sum(prices[neighbor_id] * (weight + 1) for neighbor_id, weight in weights) / total_weight
            knn_price_dict[item_id] = round(weighted_price, 2)
        return knn_price_dict

    @staticmethod
    def transform(
            embeddings: Dict[int, np.ndarray],
            prices: Dict[int, float],
            top: int,
    ) -> Dict[int, float]:
        """Transforming input embeddings into a dictionary
        with weighted average prices for each item.

        Args:
            embeddings (Dict[int, np.ndarray]): Items embeddings.
            prices (Dict[int, float]): Price dict for each item.
            top (int): Number of top neighbors to consider.

        Returns:
            Dict[int, float]: Dict with weighted average prices for each item.
        """
        similarity = SimilarItems.similarity(embeddings)
        knn = SimilarItems.knn(similarity, top)
        knn_price_dict = SimilarItems.knn_price(knn, prices)
        return knn_price_dict


if __name__ == "__main__":
    embeddings_example = {
        1: np.array([-26.57, -76.61, 81.61, -9.11, 74.8, 54.23, 32.56, -22.62, -72.44, -82.78]),
        2: np.array([-55.98, 82.87, 86.07, 18.71, -18.66, -46.74, -68.18, 60.29, 98.92, -78.95]),
        3: np.array([-27.97, 25.39, -96.85, 3.51, 95.57, -27.48, -80.27, 8.39, 89.96, -36.68]),
        4: np.array([-37.0, -49.39, 43.3, 73.36, 29.98, -56.44, -15.91, -56.46, 24.54, 12.43]),
        5: np.array([-22.71, 4.47, -65.42, 10.11, 98.34, 17.96, -10.77, 2.5, -26.55, 69.16])
    }

    prices_example = {
        1: 100.5,
        2: 12.2,
        3: 60.0,
        4: 11.1,
        5: 245.2
    }

    print(SimilarItems.transform(embeddings_example, prices_example, 5))

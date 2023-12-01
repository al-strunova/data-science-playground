"""
This module evaluates information retrieval models using cumulative gain metrics.
It includes functions for calculating Cumulative Gain (CG), Discounted Cumulative Gain (DCG),
and Normalized Discounted Cumulative Gain (nDCG). These functions help in assessing the
effectiveness of search algorithms and recommendation systems, considering top-k elements
in relevance scores.

Includes:
- cumulative_gain: Calculates CG@k.
- discounted_cumulative_gain: Computes DCG@k.
- normalized_dcg: Determines nDCG@k.
- avg_ndcg: Calculates average nDCG for multiple queries.

Example:
Calculate average nDCG for given relevance scores of queries, top 5 results, using 'standard' method.
"""

from typing import List

import numpy as np


def cumulative_gain(relevance: List[float], k: int) -> float:
    """Score is cumulative gain at k (CG@k)

    Parameters
    ----------
    relevance:  `List[float]`
        Relevance labels (Ranks)
    k : `int`
        Number of elements to be counted

    Returns
    -------
    score : float
    """
    relevance = np.asfarray(relevance)
    score = sum(relevance[:k])
    return score


def discounted_cumulative_gain(relevance: List[float], k: int, method: str = "standard") -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the value
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    relevance = np.asfarray(relevance[:k])
    indices = np.arange(2, k + 2)
    if method == 'standard':
        score = sum(relevance / np.log2(indices))
    elif method == 'industry':
        score = sum((np.exp2(relevance) - 1) / np.log2(indices))
    else:
        raise ValueError
    return score


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    score = discounted_cumulative_gain(relevance, k, method) / discounted_cumulative_gain(
        sorted(relevance, reverse=True), k, method)
    return score


def avg_ndcg(list_relevance: List[List[float]], k: int, method: str = 'standard') -> float:
    """average nDCG

    Parameters
    ----------
    list_relevance : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    score = sum([normalized_dcg(query, k, method) for query in list_relevance]) / len(list_relevance)
    return score


queries_relevance = [
        [0.99, 0.94, 0.88, 0.89, 0.72, 0.65],
        [0.99, 0.92, 0.93, 0.74, 0.61, 0.68],
        [0.99, 0.96, 0.81, 0.73, 0.76, 0.69]
    ]
k_top = 5
dcg_method = 'standard'
print(avg_ndcg(queries_relevance, k_top, dcg_method))

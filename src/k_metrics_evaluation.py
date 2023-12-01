"""
This module contains functions for evaluating classification models at a specified cutoff (k).
These functions include recall_at_k, precision_at_k, specificity_at_k, and f1_at_k, which
calculate the respective metrics for the top k predictions based on their scores.

Functions:
- recall_at_k: Calculates the recall at the top k predictions.
- precision_at_k: Computes the precision for the top k predictions.
- specificity_at_k: Determines the specificity for the top k predictions.
- f1_at_k: Calculates the F1 score, which is the harmonic mean of precision and recall, for the top k predictions.

These metrics are particularly useful for evaluating ranking models in information retrieval or recommendation systems.

Example:
- To evaluate a classification model, provide a list of true labels and predicted scores,
  and specify the value of k (default 5) for calculating these metrics.
"""


from typing import List


def recall_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """calculate recall for top 5"""
    sorted_labels, _ = zip(*sorted(zip(labels, scores), key=lambda x: x[1], reverse=True))
    return sum(sorted_labels[:k]) / sum(sorted_labels)


def precision_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """calculate precision for top 5"""
    sorted_labels, _ = zip(*sorted(zip(labels, scores), key=lambda x: x[1], reverse=True))
    return sum(sorted_labels[:k]) / k


def specificity_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """calculate specificity for top 5"""
    sorted_labels, _ = zip(*sorted(zip(labels, scores), key=lambda x: x[1], reverse=True))
    actual_n = sum([x == 0 for x in sorted_labels])

    if actual_n == 0:
        return 0

    return sum([x == 0 for x in sorted_labels[k:]])/actual_n


def f1_at_k(labels: List[int], scores: List[float], k=5) -> float:
    """calculate f1 for top 5"""
    precision_k = precision_at_k(labels, scores, k)
    recall_k = recall_at_k(labels, scores, k)

    if precision_k + recall_k == 0:
        return 0.0

    return 2 * precision_k * recall_k / (precision_k + recall_k)

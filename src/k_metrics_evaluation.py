"""
Greenterest Image Platform

Overview:
Greenterest stores neural network-generated images, offering a search service
that returns 20+ relevant images per query. The model measures image relevance.

Objective:
Evaluate model performance using historical user interactions and compare
new algorithm's image re-ranking against the old system.

Metrics:
1. Recall @ K: Proportion of clicked images identified as positive.
2. Precision @ K: Accuracy of positive predictions (clicked images).
3. F1-Score @ K: Harmonic mean of Precision and Recall.
4. Specificity @ K: Correct identification of true negatives.

Task:
Implement these four metrics. Focus on top-ranking images, as users typically
view only the first 10-20 results.

Note:
The model aims to ensure that the most relevant images appear in the top results.

Author: Aliaksandra Strunova
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

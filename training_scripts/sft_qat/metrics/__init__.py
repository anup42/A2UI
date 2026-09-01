"""
Metrics module for SFT evaluation.

This module contains various metrics for evaluating model predictions:
- TreeMatcher: JSON tree matching metrics (direct match, value-only match)
- MetricsAggregator: Aggregates all metrics and logs to TensorBoard
"""

from .tree_matcher import TreeMatcher, JSONMatcher, levenshtein_distance
from .aggregator import MetricsAggregator

__all__ = [
    'TreeMatcher',
    'JSONMatcher',
    'levenshtein_distance',
    'MetricsAggregator',
]
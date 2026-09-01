"""
Metrics Aggregator for SFT Evaluation.

This module provides a unified interface for computing all metrics
and logging them to TensorBoard.

Usage:
    from metrics import MetricsAggregator

    aggregator = MetricsAggregator()
    results = aggregator.compute_all_metrics(reference, prediction)
    aggregator.log_to_tensorboard(writer, step, results)
"""

import json
import os
from typing import Any, Dict, List, Optional, Union
import numpy as np

# Import from the same package
from .tree_matcher import TreeMatcher


class MetricsAggregator:
    """
    Aggregates all metrics for SFT evaluation.

    Provides methods to:
    - Compute all metrics for a single prediction
    - Compute batch metrics
    - Log metrics to TensorBoard
    """

    def __init__(
        self,
        tree_matcher_key_weight: float = 0.3,
        tree_matcher_value_weight: float = 0.7,
    ):
        """
        Initialize the metrics aggregator.

        Args:
            tree_matcher_key_weight: Weight for key similarity in tree matching
            tree_matcher_value_weight: Weight for value similarity in tree matching
        """
        # Initialize individual metrics
        self.tree_matcher = TreeMatcher(
            key_weight=tree_matcher_key_weight,
            value_weight=tree_matcher_value_weight,
        )

        # Track running statistics
        self.reset_statistics()

    def reset_statistics(self):
        """Reset all running statistics."""
        self.stats = {
            'exact_match': [],
            'valid_json': [],
            'direct_match_score': [],
            'value_only_match_score': [],
        }

    def compute_all_metrics(
        self,
        reference: str,
        prediction: str,
        compute_tree_match: bool = True,
    ) -> Dict:
        """
        Compute all metrics for a single reference-prediction pair.

        Args:
            reference: Reference JSON string
            prediction: Predicted JSON string
            compute_tree_match: Whether to compute tree matching metrics

        Returns:
            Dictionary with all computed metrics
        """
        results = {
            'reference': reference,
            'prediction': prediction,
        }

        # Basic metrics
        results['exact_match'] = self._compute_exact_match(reference, prediction)
        results['valid_reference_json'] = self._is_valid_json(reference)
        results['valid_prediction_json'] = self._is_valid_json(prediction)

        # Tree matching metrics
        if compute_tree_match:
            tree_results = self.tree_matcher.compute_metrics(reference, prediction)
            results['tree_match'] = tree_results
            results['direct_match_score'] = tree_results.get('direct_match_score', 0.0)
            results['value_only_match_score'] = tree_results.get('value_only_match_score', 0.0)
        else:
            results['direct_match_score'] = 0.0
            results['value_only_match_score'] = 0.0

        return results

    def compute_batch_metrics(
        self,
        references: List[str],
        predictions: List[str],
        compute_tree_match: bool = True,
    ) -> Dict:
        """
        Compute metrics for a batch of reference-prediction pairs.

        Args:
            references: List of reference JSON strings
            predictions: List of predicted JSON strings
            compute_tree_match: Whether to compute tree matching metrics

        Returns:
            Dictionary with aggregated metrics
        """
        assert len(references) == len(predictions), "Mismatched number of references and predictions"

        all_results = []
        exact_matches = []
        valid_refs = []
        valid_preds = []
        direct_scores = []
        value_only_scores = []

        for ref, pred in zip(references, predictions):
            result = self.compute_all_metrics(ref, pred, compute_tree_match)
            all_results.append(result)

            exact_matches.append(result['exact_match'])
            valid_refs.append(result['valid_reference_json'])
            valid_preds.append(result['valid_prediction_json'])

            if compute_tree_match and result.get('valid_reference_json') and result.get('valid_prediction_json'):
                direct_scores.append(result['direct_match_score'])
                value_only_scores.append(result['value_only_match_score'])

        # Compute aggregated statistics
        aggregated = {
            'num_samples': len(references),
            'exact_match_accuracy': np.mean(exact_matches) if exact_matches else 0.0,
            'valid_reference_json_ratio': np.mean(valid_refs) if valid_refs else 0.0,
            'valid_prediction_json_ratio': np.mean(valid_preds) if valid_preds else 0.0,
            'mean_direct_match_score': np.mean(direct_scores) if direct_scores else 0.0,
            'mean_value_only_match_score': np.mean(value_only_scores) if value_only_scores else 0.0,
            'std_direct_match_score': np.std(direct_scores) if direct_scores else 0.0,
            'std_value_only_match_score': np.std(value_only_scores) if value_only_scores else 0.0,
            'individual_results': all_results,
        }

        # Update running statistics
        self.stats['exact_match'].extend(exact_matches)
        self.stats['valid_json'].extend(valid_preds)
        self.stats['direct_match_score'].extend(direct_scores)
        self.stats['value_only_match_score'].extend(value_only_scores)

        return aggregated

    def log_to_tensorboard(
        self,
        writer,  # SummaryWriter from torch.utils.tensorboard
        step: int,
        metrics: Dict,
        prefix: str = "",
    ):
        """
        Log metrics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter instance
            step: Current step/epoch
            metrics: Dictionary of metrics to log
            prefix: Optional prefix for metric names
        """
        if writer is None:
            return

        # Log scalar metrics
        scalar_metrics = [
            ('exact_match_accuracy', metrics.get('exact_match_accuracy', 0)),
            ('valid_reference_json_ratio', metrics.get('valid_reference_json_ratio', 0)),
            ('valid_prediction_json_ratio', metrics.get('valid_prediction_json_ratio', 0)),
            ('mean_direct_match_score', metrics.get('mean_direct_match_score', 0)),
            ('mean_value_only_match_score', metrics.get('mean_value_only_match_score', 0)),
            ('std_direct_match_score', metrics.get('std_direct_match_score', 0)),
            ('std_value_only_match_score', metrics.get('std_value_only_match_score', 0)),
        ]

        for name, value in scalar_metrics:
            metric_name = f"{prefix}/{name}" if prefix else name
            writer.add_scalar(metric_name, value, step)

        # Log histograms for score distributions
        if self.stats['direct_match_score']:
            writer.add_histogram(
                f"{prefix}/direct_match_score_distribution" if prefix else "direct_match_score_distribution",
                np.array(self.stats['direct_match_score']),
                step,
            )

        if self.stats['value_only_match_score']:
            writer.add_histogram(
                f"{prefix}/value_only_match_score_distribution" if prefix else "value_only_match_score_distribution",
                np.array(self.stats['value_only_match_score']),
                step,
            )

        writer.flush()

    def log_to_tensorboard_single(
        self,
        writer,  # SummaryWriter from torch.utils.tensorboard
        step: int,
        metrics: Dict,
        prefix: str = "",
    ):
        """
        Log single sample metrics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter instance
            step: Current step
            metrics: Dictionary of metrics for a single sample
            prefix: Optional prefix for metric names
        """
        if writer is None:
            return

        # Log scalar metrics
        scalar_metrics = [
            ('exact_match', metrics.get('exact_match', 0)),
            ('valid_reference_json', metrics.get('valid_reference_json', 0)),
            ('valid_prediction_json', metrics.get('valid_prediction_json', 0)),
            ('direct_match_score', metrics.get('direct_match_score', 0)),
            ('value_only_match_score', metrics.get('value_only_match_score', 0)),
        ]

        for name, value in scalar_metrics:
            metric_name = f"{prefix}/{name}" if prefix else name
            writer.add_scalar(metric_name, value, step)

        writer.flush()

    def get_summary(self) -> Dict:
        """
        Get summary statistics of all computed metrics.

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_samples': len(self.stats['exact_match']),
            'exact_match_accuracy': np.mean(self.stats['exact_match']) if self.stats['exact_match'] else 0.0,
            'valid_json_ratio': np.mean(self.stats['valid_json']) if self.stats['valid_json'] else 0.0,
            'mean_direct_match_score': np.mean(self.stats['direct_match_score']) if self.stats['direct_match_score'] else 0.0,
            'mean_value_only_match_score': np.mean(self.stats['value_only_match_score']) if self.stats['value_only_match_score'] else 0.0,
            'std_direct_match_score': np.std(self.stats['direct_match_score']) if self.stats['direct_match_score'] else 0.0,
            'std_value_only_match_score': np.std(self.stats['value_only_match_score']) if self.stats['value_only_match_score'] else 0.0,
        }

        return summary

    def save_summary(self, output_path: str):
        """
        Save summary statistics to a JSON file.

        Args:
            output_path: Path to save the summary JSON
        """
        summary = self.get_summary()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # ==================== Helper Methods ====================

    def _compute_exact_match(self, reference: str, prediction: str) -> bool:
        """Compute exact match between reference and prediction."""
        return reference.strip() == prediction.strip()

    def _is_valid_json(self, text: str) -> bool:
        """Check if text is valid JSON."""
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, TypeError):
            return False


# ==================== Convenience Functions ====================

def compute_metrics(
    references: List[str],
    predictions: List[str],
    output_dir: Optional[str] = None,
    tensorboard_writer=None,
    step: int = 0,
    prefix: str = "eval",
) -> Dict:
    """
    Convenience function to compute all metrics and optionally log to TensorBoard.

    Args:
        references: List of reference JSON strings
        predictions: List of predicted JSON strings
        output_dir: Optional directory to save summary
        tensorboard_writer: Optional TensorBoard writer
        step: Current step for TensorBoard logging
        prefix: Prefix for TensorBoard metrics

    Returns:
        Dictionary with aggregated metrics
    """
    aggregator = MetricsAggregator()
    results = aggregator.compute_batch_metrics(references, predictions)

    # Log to TensorBoard
    if tensorboard_writer is not None:
        aggregator.log_to_tensorboard(tensorboard_writer, step, results, prefix)

    # Save summary
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "metrics_summary.json")
        aggregator.save_summary(summary_path)

    return results
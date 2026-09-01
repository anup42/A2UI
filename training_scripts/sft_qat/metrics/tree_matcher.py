"""
Tree Matching Metrics for JSON Comparison.

Provides two matching modes:
1. Direct Match: Match key-value pairs (considers both keys and values)
2. Value-Only Match: Match values only (ignores keys)
"""

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


class JSONMatcher:
    """
    Two-mode JSON matching algorithm:
    1. Direct Match: Match key-value pairs
    2. Value-Only Match: Match values ignoring keys
    """

    def __init__(self, key_weight: float = 0.3, value_weight: float = 0.7):
        """
        Initialize the JSON matcher.

        Args:
            key_weight: Weight for key similarity in direct match mode
            value_weight: Weight for value similarity in direct match mode
        """
        self.key_weight = key_weight
        self.value_weight = value_weight

    def flatten_json(self, obj: Any, prefix: str = "", sep: str = ".") -> Dict[str, Any]:
        """
        Flatten nested JSON to dot-notation keys.

        Args:
            obj: JSON object to flatten
            prefix: Current key prefix
            sep: Separator for nested keys

        Returns:
            Flattened dictionary with dot-notation keys
        """
        items = {}

        if isinstance(obj, dict):
            for k, v in obj.items():
                new_key = f"{prefix}{sep}{k}" if prefix else k
                items.update(self.flatten_json(v, new_key, sep))
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                new_key = f"{prefix}{sep}{i}" if prefix else str(i)
                items.update(self.flatten_json(v, new_key, sep))
        else:
            items[prefix] = obj

        return items

    def extract_all_values(self, obj: Any) -> List[Any]:
        """
        Extract all leaf values from JSON, ignoring keys.

        Args:
            obj: JSON object to extract values from

        Returns:
            List of all leaf values
        """
        values = []

        if isinstance(obj, dict):
            for v in obj.values():
                values.extend(self.extract_all_values(v))
        elif isinstance(obj, list):
            for v in obj:
                values.extend(self.extract_all_values(v))
        else:
            values.append(obj)

        return values

    def key_similarity(self, key1: str, key2: str) -> float:
        """
        Compute similarity between two keys.

        Args:
            key1: First key
            key2: Second key

        Returns:
            Similarity score in [0, 1]
        """
        # Normalize keys
        k1 = key1.lower().replace("_", "").replace("-", "")
        k2 = key2.lower().replace("_", "").replace("-", "")

        # Exact match
        if k1 == k2:
            return 1.0

        # Levenshtein-based similarity
        dist = levenshtein_distance(k1, k2)
        max_len = max(len(k1), len(k2))
        return 1.0 - (dist / max_len) if max_len > 0 else 1.0

    def value_similarity(self, val1: Any, val2: Any) -> float:
        """
        Compute similarity between two values.

        Args:
            val1: First value
            val2: Second value

        Returns:
            Similarity score in [0, 1]
        """
        s1, s2 = str(val1), str(val2)

        # Exact match
        if s1 == s2:
            return 1.0

        # Handle numeric values
        try:
            n1, n2 = float(s1), float(s2)
            # float() parses "nan"/"inf"/"-infinity" without raising -- an
            # undertrained model can emit these as literal token strings, and an
            # overflowing value (e.g. "1e400") also lands here as inf. Treat
            # either as a non-numeric mismatch (fall through to Levenshtein)
            # rather than let a NaN/Inf leak into the Hungarian cost matrix,
            # where scipy raises "matrix contains invalid numeric entries" and
            # kills the whole eval run.
            if math.isfinite(n1) and math.isfinite(n2):
                if n1 == 0 and n2 == 0:
                    return 1.0
                similarity = 1.0 - abs(n1 - n2) / (abs(n1) + abs(n2) + 1e-10)
                # Guard against overflow in the subtraction/sum above even when
                # n1, n2 were individually finite (e.g. both near float max).
                if math.isfinite(similarity):
                    return similarity
        except (ValueError, TypeError):
            pass

        # Levenshtein similarity for strings
        dist = levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (dist / max_len) if max_len > 0 else 1.0

    def direct_match(self, json1: Dict, json2: Dict) -> Dict:
        """
        Match key-value pairs from both JSONs.

        Args:
            json1: First JSON object
            json2: Second JSON object

        Returns:
            Dictionary with match results including score and matched pairs
        """
        # Step 1: Flatten JSONs
        flat1 = self.flatten_json(json1, prefix="")
        flat2 = self.flatten_json(json2, prefix="")

        if not flat1 or not flat2:
            return {
                'mode': 'direct_match',
                'score': 0.0,
                'matched_pairs': [],
                'unmatched_in_json1': list(flat1.keys()),
                'unmatched_in_json2': list(flat2.keys()),
                'total_pairs_json1': len(flat1),
                'total_pairs_json2': len(flat2),
            }

        # Step 2: Compute similarity matrix
        keys1, keys2 = list(flat1.keys()), list(flat2.keys())
        values1, values2 = list(flat1.values()), list(flat2.values())

        n, m = len(keys1), len(keys2)
        cost_matrix = np.zeros((n, m))
        similarity_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                key_sim = self.key_similarity(keys1[i], keys2[j])
                value_sim = self.value_similarity(values1[i], values2[j])
                combined = self.key_weight * key_sim + self.value_weight * value_sim
                similarity_matrix[i, j] = combined
                cost_matrix[i, j] = -combined  # Negative for Hungarian (minimization)

        # Step 3: Hungarian algorithm for optimal matching
        # Belt-and-suspenders: value_similarity() already guards NaN/Inf inputs, but
        # scipy raises "matrix contains invalid numeric entries" on ANY non-finite
        # cell and aborts the whole eval run -- so also sanitize here in case a
        # future similarity function reintroduces one. Worst score (1.0, i.e.
        # maximally dissimilar after negation) rather than crashing.
        if not np.all(np.isfinite(cost_matrix)):
            cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=-1.0)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Step 4: Compute results
        matched_pairs = []
        total_score = 0

        matched_indices_1 = set(row_ind)
        matched_indices_2 = set(col_ind)

        for i, j in zip(row_ind, col_ind):
            key_sim = self.key_similarity(keys1[i], keys2[j])
            value_sim = self.value_similarity(values1[i], values2[j])
            combined = similarity_matrix[i, j]
            matched_pairs.append({
                'key1': keys1[i],
                'key2': keys2[j],
                'value1': values1[i],
                'value2': values2[j],
                'key_similarity': round(key_sim, 4),
                'value_similarity': round(value_sim, 4),
                'combined_score': round(combined, 4),
            })
            total_score += combined

        # Normalize score
        max_possible = max(n, m)
        normalized_score = total_score / max_possible if max_possible > 0 else 0

        return {
            'mode': 'direct_match',
            'score': round(normalized_score, 4),
            'matched_pairs': matched_pairs,
            'unmatched_in_json1': [keys1[i] for i in range(n) if i not in matched_indices_1],
            'unmatched_in_json2': [keys2[j] for j in range(m) if j not in matched_indices_2],
            'total_pairs_json1': n,
            'total_pairs_json2': m,
        }

    def value_only_match(self, json1: Dict, json2: Dict) -> Dict:
        """
        Match values only, completely ignoring keys.

        Args:
            json1: First JSON object
            json2: Second JSON object

        Returns:
            Dictionary with match results including score and matched values
        """
        # Step 1: Extract all values
        values1 = self.extract_all_values(json1)
        values2 = self.extract_all_values(json2)

        if not values1 or not values2:
            return {
                'mode': 'value_only_match',
                'score': 0.0,
                'matched_values': [],
                'unmatched_in_json1': values1,
                'unmatched_in_json2': values2,
                'total_values_json1': len(values1),
                'total_values_json2': len(values2),
            }

        # Step 2: Compute similarity matrix
        n, m = len(values1), len(values2)
        cost_matrix = np.zeros((n, m))
        similarity_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                sim = self.value_similarity(values1[i], values2[j])
                similarity_matrix[i, j] = sim
                cost_matrix[i, j] = -sim

        # Step 3: Hungarian algorithm
        # Same guard as direct_match(): sanitize in case value_similarity() is
        # ever bypassed or reintroduces a NaN/Inf -- scipy aborts the whole eval
        # run on any non-finite cell otherwise.
        if not np.all(np.isfinite(cost_matrix)):
            cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=-1.0)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Step 4: Compute results
        matched_values = []
        total_score = 0

        matched_indices_1 = set(row_ind)
        matched_indices_2 = set(col_ind)

        for i, j in zip(row_ind, col_ind):
            sim = similarity_matrix[i, j]
            matched_values.append({
                'value1': values1[i],
                'value2': values2[j],
                'similarity': round(sim, 4),
            })
            total_score += sim

        # Normalize
        max_possible = max(n, m)
        normalized_score = total_score / max_possible if max_possible > 0 else 0

        return {
            'mode': 'value_only_match',
            'score': round(normalized_score, 4),
            'matched_values': matched_values,
            'unmatched_in_json1': [values1[i] for i in range(n) if i not in matched_indices_1],
            'unmatched_in_json2': [values2[j] for j in range(m) if j not in matched_indices_2],
            'total_values_json1': n,
            'total_values_json2': m,
        }

    def compare(self, json1: Dict, json2: Dict, mode: str = 'both') -> Dict:
        """
        Compare two JSONs using specified mode.

        Args:
            json1: First JSON object (reference)
            json2: Second JSON object (prediction)
            mode: 'direct', 'value_only', or 'both'

        Returns:
            Dictionary with comparison results
        """
        results = {
            'reference_type': type(json1).__name__,
            'prediction_type': type(json2).__name__,
        }

        if mode in ['direct', 'both']:
            results['direct_match'] = self.direct_match(json1, json2)

        if mode in ['value_only', 'both']:
            results['value_only_match'] = self.value_only_match(json1, json2)

        # Summary
        results['summary'] = self._create_summary(results)

        return results

    def _create_summary(self, results: Dict) -> Dict:
        """Create a summary of the comparison results."""
        direct_score = results.get('direct_match', {}).get('score', 0)
        value_only_score = results.get('value_only_match', {}).get('score', 0)

        return {
            'direct_match_score': direct_score,
            'value_only_score': value_only_score,
            'interpretation': self._interpret_results(direct_score, value_only_score),
        }

    def _interpret_results(self, direct_score: float, value_only_score: float) -> str:
        """Interpret the comparison results."""
        if direct_score > 0.8 and value_only_score > 0.8:
            return "Highly similar: Both structure and values match well"
        elif direct_score < 0.5 and value_only_score > 0.8:
            return "Different structure, similar content: Keys differ but values match"
        elif direct_score > 0.8 and value_only_score < 0.5:
            return "Similar structure, different content: Keys match but values differ"
        elif direct_score < 0.3 and value_only_score < 0.3:
            return "Low similarity: Both structure and content differ"
        else:
            return "Moderate similarity: Partial match in structure and/or content"


class TreeMatcher:
    """
    High-level tree matching class for use in evaluation pipelines.
    Provides easy-to-use methods for computing tree edit distance metrics.
    """

    def __init__(self, key_weight: float = 0.3, value_weight: float = 0.7):
        """
        Initialize the tree matcher.

        Args:
            key_weight: Weight for key similarity
            value_weight: Weight for value similarity
        """
        self.matcher = JSONMatcher(key_weight=key_weight, value_weight=value_weight)

    def compute_metrics(self, reference: str, prediction: str) -> Dict:
        """
        Compute all tree matching metrics between reference and prediction.

        Args:
            reference: Reference JSON string
            prediction: Predicted JSON string

        Returns:
            Dictionary with all metrics
        """
        # Parse JSONs
        try:
            ref_json = json.loads(reference) if isinstance(reference, str) else reference
        except (json.JSONDecodeError, TypeError) as e:
            return {
                'error': f"Failed to parse reference JSON: {str(e)}",
                'direct_match_score': 0.0,
                'value_only_match_score': 0.0,
                'valid_reference_json': False,
            }

        try:
            pred_json = json.loads(prediction) if isinstance(prediction, str) else prediction
        except (json.JSONDecodeError, TypeError) as e:
            return {
                'error': f"Failed to parse prediction JSON: {str(e)}",
                'direct_match_score': 0.0,
                'value_only_match_score': 0.0,
                'valid_prediction_json': False,
            }

        # Compute comparison
        results = self.matcher.compare(ref_json, pred_json, mode='both')

        return {
            'direct_match_score': results['direct_match']['score'],
            'value_only_match_score': results['value_only_match']['score'],
            'direct_match_details': results['direct_match'],
            'value_only_match_details': results['value_only_match'],
            'summary': results['summary'],
            'valid_reference_json': True,
            'valid_prediction_json': True,
        }

    def compute_batch_metrics(self, references: List[str], predictions: List[str]) -> Dict:
        """
        Compute metrics for a batch of reference-prediction pairs.

        Args:
            references: List of reference JSON strings
            predictions: List of predicted JSON strings

        Returns:
            Dictionary with aggregated metrics
        """
        all_results = []
        direct_scores = []
        value_only_scores = []
        valid_json_count = 0

        for ref, pred in zip(references, predictions):
            result = self.compute_metrics(ref, pred)
            all_results.append(result)

            if result.get('valid_reference_json', False) and result.get('valid_prediction_json', False):
                direct_scores.append(result['direct_match_score'])
                value_only_scores.append(result['value_only_match_score'])
                valid_json_count += 1

        return {
            'individual_results': all_results,
            'mean_direct_match_score': np.mean(direct_scores) if direct_scores else 0.0,
            'mean_value_only_match_score': np.mean(value_only_scores) if value_only_scores else 0.0,
            'std_direct_match_score': np.std(direct_scores) if direct_scores else 0.0,
            'std_value_only_match_score': np.std(value_only_scores) if value_only_scores else 0.0,
            'valid_json_count': valid_json_count,
            'total_count': len(references),
        }
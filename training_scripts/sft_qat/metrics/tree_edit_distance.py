"""
Tree Edit Distance Calculator for JSON-based UI Component Trees.

This module provides tree edit distance calculation using the Zhang-Shasha algorithm,
integrated with the metrics pipeline for SFT evaluation.

The tree edit distance measures the minimum number of operations (insert, delete, relabel)
needed to transform one tree into another, providing a structural similarity metric.
"""

import json
import sys
import os
from typing import Dict, List, Optional, Any, Tuple

# Add parent directory to path to import tree_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tree_utils import (
    TreeNode,
    json_to_tree,
    tree_edit_distance as compute_tree_edit_distance,
    simple_tree_distance,
    tree_to_string,
)


class TreeEditDistanceCalculator:
    """
    Calculator for tree edit distance metrics between JSON structures.

    This class provides methods to:
    - Parse JSON strings into tree structures
    - Compute tree edit distance using Zhang-Shasha algorithm
    - Compute normalized tree edit distance (similarity score)
    - Batch process multiple reference-prediction pairs
    """

    def __init__(
        self,
        cost_insert: int = 1,
        cost_delete: int = 1,
        cost_relabel: int = 1,
        use_simple_method: bool = True,
    ):
        """
        Initialize the tree edit distance calculator.

        Args:
            cost_insert: Cost of inserting a node
            cost_delete: Cost of deleting a node
            cost_relabel: Cost of relabeling a node
            use_simple_method: If True, use simple recursive method (more reliable for complex trees)
                              If False, use Zhang-Shasha algorithm
        """
        self.cost_insert = cost_insert
        self.cost_delete = cost_delete
        self.cost_relabel = cost_relabel
        self.use_simple_method = use_simple_method

    def parse_json_to_tree(self, json_str: str) -> Optional[TreeNode]:
        """
        Parse a JSON string into a TreeNode structure.

        Args:
            json_str: JSON string to parse

        Returns:
            Root TreeNode of the parsed tree, or None if parsing fails
        """
        try:
            # Parse JSON string
            if isinstance(json_str, str):
                data = json.loads(json_str)
            else:
                data = json_str

            # Convert to tree structure
            return json_to_tree(data)
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            return None

    def compute_distance(
        self,
        tree1: Optional[TreeNode],
        tree2: Optional[TreeNode],
    ) -> int:
        """
        Compute the tree edit distance between two trees.

        Args:
            tree1: First tree (reference)
            tree2: Second tree (prediction)

        Returns:
            Tree edit distance (number of operations)
        """
        if tree1 is None and tree2 is None:
            return 0
        if tree1 is None:
            # Need to insert all nodes from tree2
            return self._count_nodes(tree2) * self.cost_insert
        if tree2 is None:
            # Need to delete all nodes from tree1
            return self._count_nodes(tree1) * self.cost_delete

        if self.use_simple_method:
            return simple_tree_distance(tree1, tree2)
        else:
            return compute_tree_edit_distance(
                tree1, tree2,
                self.cost_insert, self.cost_delete, self.cost_relabel
            )

    def _count_nodes(self, node: TreeNode) -> int:
        """Count total nodes in a tree."""
        if node is None:
            return 0
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count

    def compute_normalized_distance(
        self,
        tree1: Optional[TreeNode],
        tree2: Optional[TreeNode],
    ) -> float:
        """
        Compute normalized tree edit distance (0 to 1).

        Normalized distance = edit_distance / max_possible_distance
        where max_possible_distance = max(nodes_in_tree1, nodes_in_tree2)

        Args:
            tree1: First tree (reference)
            tree2: Second tree (prediction)

        Returns:
            Normalized distance in range [0, 1] where 0 means identical
        """
        if tree1 is None and tree2 is None:
            return 0.0

        nodes1 = self._count_nodes(tree1) if tree1 else 0
        nodes2 = self._count_nodes(tree2) if tree2 else 0

        if nodes1 == 0 and nodes2 == 0:
            return 0.0

        max_possible = max(nodes1, nodes2)
        distance = self.compute_distance(tree1, tree2)

        return distance / max_possible if max_possible > 0 else 0.0

    def compute_similarity_score(
        self,
        tree1: Optional[TreeNode],
        tree2: Optional[TreeNode],
    ) -> float:
        """
        Compute tree similarity score (0 to 1).

        Similarity = 1 - normalized_distance

        Args:
            tree1: First tree (reference)
            tree2: Second tree (prediction)

        Returns:
            Similarity score in range [0, 1] where 1 means identical
        """
        return 1.0 - self.compute_normalized_distance(tree1, tree2)

    def compute_metrics(
        self,
        reference: str,
        prediction: str,
    ) -> Dict:
        """
        Compute all tree edit distance metrics for a reference-prediction pair.

        Args:
            reference: Reference JSON string
            prediction: Predicted JSON string

        Returns:
            Dictionary with tree edit distance metrics
        """
        result = {
            'valid_reference_json': False,
            'valid_prediction_json': False,
            'tree_edit_distance': None,
            'normalized_tree_edit_distance': None,
            'tree_similarity_score': None,
            'reference_node_count': None,
            'prediction_node_count': None,
        }

        # Parse reference
        try:
            ref_tree = self.parse_json_to_tree(reference)
            result['valid_reference_json'] = ref_tree is not None
            result['reference_node_count'] = self._count_nodes(ref_tree) if ref_tree else 0
        except Exception as e:
            ref_tree = None
            result['reference_parse_error'] = str(e)

        # Parse prediction
        try:
            pred_tree = self.parse_json_to_tree(prediction)
            result['valid_prediction_json'] = pred_tree is not None
            result['prediction_node_count'] = self._count_nodes(pred_tree) if pred_tree else 0
        except Exception as e:
            pred_tree = None
            result['prediction_parse_error'] = str(e)

        # Compute distance if both are valid
        if ref_tree is not None and pred_tree is not None:
            distance = self.compute_distance(ref_tree, pred_tree)
            normalized = self.compute_normalized_distance(ref_tree, pred_tree)
            similarity = self.compute_similarity_score(ref_tree, pred_tree)

            result['tree_edit_distance'] = distance
            result['normalized_tree_edit_distance'] = normalized
            result['tree_similarity_score'] = similarity
        elif ref_tree is None and pred_tree is None:
            # Both invalid - consider as matching (both failed)
            result['tree_edit_distance'] = 0
            result['normalized_tree_edit_distance'] = 0.0
            result['tree_similarity_score'] = 1.0
        else:
            # One valid, one invalid - max distance
            max_nodes = max(
                result['reference_node_count'] or 0,
                result['prediction_node_count'] or 0
            )
            result['tree_edit_distance'] = max_nodes
            result['normalized_tree_edit_distance'] = 1.0
            result['tree_similarity_score'] = 0.0

        return result

    def compute_batch_metrics(
        self,
        references: List[str],
        predictions: List[str],
    ) -> Dict:
        """
        Compute tree edit distance metrics for a batch of reference-prediction pairs.

        Args:
            references: List of reference JSON strings
            predictions: List of predicted JSON strings

        Returns:
            Dictionary with aggregated metrics
        """
        assert len(references) == len(predictions), "Mismatched number of references and predictions"

        all_results = []
        distances = []
        normalized_distances = []
        similarity_scores = []
        valid_pairs = []

        for ref, pred in zip(references, predictions):
            result = self.compute_metrics(ref, pred)
            all_results.append(result)

            if result['valid_reference_json'] and result['valid_prediction_json']:
                distances.append(result['tree_edit_distance'])
                normalized_distances.append(result['normalized_tree_edit_distance'])
                similarity_scores.append(result['tree_similarity_score'])
                valid_pairs.append(True)
            else:
                valid_pairs.append(False)

        # Compute statistics
        import numpy as np

        return {
            'num_samples': len(references),
            'valid_pairs_count': sum(valid_pairs),
            'mean_tree_edit_distance': np.mean(distances) if distances else 0.0,
            'std_tree_edit_distance': np.std(distances) if distances else 0.0,
            'mean_normalized_tree_edit_distance': np.mean(normalized_distances) if normalized_distances else 0.0,
            'std_normalized_tree_edit_distance': np.std(normalized_distances) if normalized_distances else 0.0,
            'mean_tree_similarity_score': np.mean(similarity_scores) if similarity_scores else 0.0,
            'std_tree_similarity_score': np.std(similarity_scores) if similarity_scores else 0.0,
            'individual_results': all_results,
        }


# Convenience function for quick usage
def compute_tree_edit_distance_metrics(
    reference: str,
    prediction: str,
    use_simple_method: bool = True,
) -> Dict:
    """
    Convenience function to compute tree edit distance metrics.

    Args:
        reference: Reference JSON string
        prediction: Predicted JSON string
        use_simple_method: If True, use simple recursive method

    Returns:
        Dictionary with tree edit distance metrics
    """
    calculator = TreeEditDistanceCalculator(use_simple_method=use_simple_method)
    return calculator.compute_metrics(reference, prediction)


if __name__ == "__main__":
    # Example usage
    example_ref = {
        "root": "root",
        "elements": {
            "root": {
                "type": "Stack",
                "props": {"direction": "vertical"},
                "children": ["header", "content"]
            },
            "header": {
                "type": "Text",
                "props": {"text": "Title"},
                "children": []
            },
            "content": {
                "type": "Card",
                "props": {},
                "children": []
            }
        }
    }

    example_pred = {
        "root": "root",
        "elements": {
            "root": {
                "type": "Stack",
                "props": {"direction": "horizontal"},
                "children": ["header", "footer"]
            },
            "header": {
                "type": "Text",
                "props": {"text": "Title"},
                "children": []
            },
            "footer": {
                "type": "Text",
                "props": {"text": "Footer"},
                "children": []
            }
        }
    }

    calculator = TreeEditDistanceCalculator()

    print("=" * 60)
    print("Tree Edit Distance Example")
    print("=" * 60)

    ref_tree = calculator.parse_json_to_tree(json.dumps(example_ref))
    pred_tree = calculator.parse_json_to_tree(json.dumps(example_pred))

    print("\nReference Tree:")
    print(tree_to_string(ref_tree) if ref_tree else "Failed to parse")

    print("\nPrediction Tree:")
    print(tree_to_string(pred_tree) if pred_tree else "Failed to parse")

    metrics = calculator.compute_metrics(json.dumps(example_ref), json.dumps(example_pred))

    print("\nMetrics:")
    print(f"  Tree Edit Distance: {metrics['tree_edit_distance']}")
    print(f"  Normalized Distance: {metrics['normalized_tree_edit_distance']:.4f}")
    print(f"  Similarity Score:    {metrics['tree_similarity_score']:.4f}")

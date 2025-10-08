"""Unit tests for two-pass refinement architecture.

Tests the core helper functions and data structures used in the two-pass
cluster refinement system.
"""

import pytest
from pathlib import Path
from typing import Dict, List

from gaphack.cluster_refinement import (
    build_refinement_scope,
    execute_refinement_operations,
    check_full_set_equivalence,
    RefinementConfig
)
from gaphack.cluster_graph import ClusterGraph


class TestFullSetEquivalence:
    """Test full cluster set equivalence checking."""

    def test_identical_sets_are_equivalent(self):
        """Test that identical cluster sets are equivalent."""
        clusters1 = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        clusters2 = {
            'cluster_X': ['seq1', 'seq2'],
            'cluster_Y': ['seq3', 'seq4']
        }

        assert check_full_set_equivalence(clusters1, clusters2)

    def test_different_sets_not_equivalent(self):
        """Test that different cluster sets are not equivalent."""
        clusters1 = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        clusters2 = {
            'cluster_merged': ['seq1', 'seq2', 'seq3', 'seq4']
        }

        assert not check_full_set_equivalence(clusters1, clusters2)

    def test_order_independent_full_equivalence(self):
        """Test that full set equivalence is order-independent."""
        clusters1 = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        clusters2 = {
            'cluster_Y': ['seq4', 'seq3'],  # Different order
            'cluster_X': ['seq2', 'seq1']   # Different order
        }

        assert check_full_set_equivalence(clusters1, clusters2)


class TestExecuteRefinementOperations:
    """Test execution of refinement operations with overlap handling."""

    def test_non_overlapping_operations(self):
        """Test that non-overlapping operations with actual changes are all applied."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4'],
            'cluster_C': ['seq5', 'seq6']
        }

        # Two non-overlapping operations that actually change the clusters
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {
                    'cluster_A_split1': ['seq1'],
                    'cluster_A_split2': ['seq2']
                },  # Split cluster A
                'scope_signature': frozenset(['seq1', 'seq2']),
                'ami': 0.5  # Changed (split)
            },
            {
                'seed_id': 'cluster_C',
                'input_cluster_ids': {'cluster_C'},
                'output_clusters': {
                    'cluster_C_split1': ['seq5'],
                    'cluster_C_split2': ['seq6']
                },  # Split cluster C
                'scope_signature': frozenset(['seq5', 'seq6']),
                'ami': 0.5  # Changed (split)
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations
        )

        # Both operations should be applied
        assert 'cluster_A_split1' in next_clusters
        assert 'cluster_A_split2' in next_clusters
        assert 'cluster_C_split1' in next_clusters
        assert 'cluster_C_split2' in next_clusters
        assert 'cluster_B' in next_clusters  # Unchanged cluster preserved
        assert len(next_clusters) == 5
        assert changes_made, "Changes should be recorded"

    def test_overlapping_operations_skip_second(self):
        """Test that overlapping operations skip the second one."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        # Two operations that try to modify cluster_A (both split it)
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {
                    'cluster_A_v1_1': ['seq1'],
                    'cluster_A_v1_2': ['seq2']
                },
                'scope_signature': frozenset(['seq1', 'seq2']),
                'ami': 0.5  # Changed
            },
            {
                'seed_id': 'cluster_A',  # Same cluster (but already consumed by first op)
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {
                    'cluster_A_v2_1': ['seq1'],
                    'cluster_A_v2_2': ['seq2']
                },
                'scope_signature': frozenset(['seq1', 'seq2']),
                'ami': 0.5  # Changed
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations
        )

        # First operation applied, second skipped because cluster_A was already consumed
        assert 'cluster_A_v1_1' in next_clusters
        assert 'cluster_A_v1_2' in next_clusters
        assert 'cluster_A_v2_1' not in next_clusters
        assert 'cluster_A_v2_2' not in next_clusters
        assert 'cluster_B' in next_clusters
        assert len(next_clusters) == 3  # 2 from split A + cluster_B

    def test_converged_scope_detected(self):
        """Test that converged scopes are detected."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        # Operation that produces identical output (converged, AMI=1.0)
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {'cluster_A_same': ['seq1', 'seq2']},  # Same content
                'scope_signature': frozenset(['seq1', 'seq2']),
                'ami': 1.0  # Perfect agreement (converged)
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations
        )

        # Should detect convergence
        assert len(converged) == 1
        assert frozenset(['seq1', 'seq2']) in converged
        assert not changes_made, "No changes should be recorded for converged scope"

    def test_merge_operation_marks_changes(self):
        """Test that merge operations are marked as changes."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        # Operation that merges two clusters
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A', 'cluster_B'},
                'output_clusters': {'cluster_merged': ['seq1', 'seq2', 'seq3', 'seq4']},
                'scope_signature': frozenset(['seq1', 'seq2', 'seq3', 'seq4']),
                'ami': 0.7  # Changed (merged)
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations
        )

        # Should mark as changed
        assert changes_made, "Merge should be marked as a change"
        assert len(converged) == 0, "Merge should not be marked as converged"
        assert 'cluster_merged' in next_clusters
        assert len(next_clusters) == 1


class TestEdgeCases:
    """Test edge cases for two-pass refinement."""

    def test_empty_clusters_input(self):
        """Test that empty cluster dict is handled gracefully."""
        clusters = {}
        assert check_full_set_equivalence(clusters, {})

    def test_single_cluster_input(self):
        """Test refinement with only one cluster."""
        clusters = {
            'cluster_1': ['seq1', 'seq2', 'seq3']
        }

        # Should be equivalent to itself
        assert check_full_set_equivalence(clusters, clusters)

    def test_all_singleton_clusters(self):
        """Test refinement with all singleton clusters."""
        clusters = {
            f'cluster_{i}': [f'seq{i}']
            for i in range(100)
        }

        # Should be equivalent to itself
        assert check_full_set_equivalence(clusters, clusters)

    def test_immediate_convergence_detected(self):
        """Test that immediate convergence (no changes) is detected."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        # Operations that produce identical output (converged immediately)
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {'cluster_A_same': ['seq1', 'seq2']},
                'scope_signature': frozenset(['seq1', 'seq2']),
                'ami': 1.0  # Perfect agreement
            },
            {
                'seed_id': 'cluster_B',
                'input_cluster_ids': {'cluster_B'},
                'output_clusters': {'cluster_B_same': ['seq3', 'seq4']},
                'scope_signature': frozenset(['seq3', 'seq4']),
                'ami': 1.0  # Perfect agreement
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations
        )

        # Both should be detected as converged
        assert len(converged) == 2
        assert not changes_made

    def test_operations_with_no_input_clusters(self):
        """Test operations that reference non-existent clusters are skipped."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        # Operation referencing cluster that doesn't exist
        operations = [
            {
                'seed_id': 'cluster_NONEXISTENT',
                'input_cluster_ids': {'cluster_NONEXISTENT'},
                'output_clusters': {'cluster_new': ['seq5', 'seq6']},
                'scope_signature': frozenset(['seq5', 'seq6']),
                'ami': 0.8  # Changed
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations
        )

        # Should skip the invalid operation and preserve original clusters
        assert 'cluster_A' in next_clusters
        assert 'cluster_B' in next_clusters
        assert 'cluster_new' not in next_clusters
        assert not changes_made


# Note: Tests for build_refinement_scope() require creating real sequences and a ClusterGraph,
# which is more complex. These would be better as integration tests or require fixtures with
# test sequence data. We'll add those in the integration test file.

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

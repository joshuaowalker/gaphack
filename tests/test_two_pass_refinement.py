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
    check_cluster_set_equivalence,
    compute_all_signatures,
    check_full_set_equivalence,
    RefinementConfig
)
from gaphack.cluster_graph import ClusterGraph


class TestComputeSignatures:
    """Test cluster signature computation for equivalence checking."""

    def test_compute_all_signatures(self):
        """Test computing frozenset signatures for all clusters."""
        clusters = {
            'cluster_A': ['seq1', 'seq2', 'seq3'],
            'cluster_B': ['seq4', 'seq5'],
            'cluster_C': ['seq6']
        }

        signatures = compute_all_signatures(clusters)

        assert len(signatures) == 3
        assert signatures['cluster_A'] == frozenset(['seq1', 'seq2', 'seq3'])
        assert signatures['cluster_B'] == frozenset(['seq4', 'seq5'])
        assert signatures['cluster_C'] == frozenset(['seq6'])

    def test_compute_signatures_preserves_order_independence(self):
        """Test that signatures are order-independent."""
        clusters1 = {
            'cluster_A': ['seq1', 'seq2', 'seq3']
        }
        clusters2 = {
            'cluster_A': ['seq3', 'seq1', 'seq2']  # Different order
        }

        sig1 = compute_all_signatures(clusters1)
        sig2 = compute_all_signatures(clusters2)

        assert sig1['cluster_A'] == sig2['cluster_A']


class TestClusterSetEquivalence:
    """Test cluster set equivalence checking."""

    def test_identical_clusters_are_equivalent(self):
        """Test that identical input/output clusters are detected as equivalent."""
        input_cluster_ids = {'cluster_A', 'cluster_B'}

        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        output_clusters = {
            'cluster_X': ['seq1', 'seq2'],
            'cluster_Y': ['seq3', 'seq4']
        }

        signatures = compute_all_signatures(current_clusters)

        is_equivalent = check_cluster_set_equivalence(
            input_cluster_ids=input_cluster_ids,
            output_clusters=output_clusters,
            current_clusters=current_clusters,
            cluster_signatures=signatures
        )

        assert is_equivalent, "Identical cluster content should be equivalent"

    def test_different_clusters_not_equivalent(self):
        """Test that different clusters are not equivalent."""
        input_cluster_ids = {'cluster_A', 'cluster_B'}

        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        # Output has merged the clusters
        output_clusters = {
            'cluster_merged': ['seq1', 'seq2', 'seq3', 'seq4']
        }

        signatures = compute_all_signatures(current_clusters)

        is_equivalent = check_cluster_set_equivalence(
            input_cluster_ids=input_cluster_ids,
            output_clusters=output_clusters,
            current_clusters=current_clusters,
            cluster_signatures=signatures
        )

        assert not is_equivalent, "Merged clusters should not be equivalent to separate clusters"

    def test_split_clusters_not_equivalent(self):
        """Test that split clusters are not equivalent."""
        input_cluster_ids = {'cluster_A'}

        current_clusters = {
            'cluster_A': ['seq1', 'seq2', 'seq3', 'seq4']
        }

        # Output has split the cluster
        output_clusters = {
            'cluster_X': ['seq1', 'seq2'],
            'cluster_Y': ['seq3', 'seq4']
        }

        signatures = compute_all_signatures(current_clusters)

        is_equivalent = check_cluster_set_equivalence(
            input_cluster_ids=input_cluster_ids,
            output_clusters=output_clusters,
            current_clusters=current_clusters,
            cluster_signatures=signatures
        )

        assert not is_equivalent, "Split cluster should not be equivalent to original"

    def test_order_independent_equivalence(self):
        """Test that equivalence is order-independent."""
        input_cluster_ids = {'cluster_A', 'cluster_B'}

        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        # Output has same clusters but different IDs and different order
        output_clusters = {
            'cluster_Y': ['seq4', 'seq3'],  # Different order
            'cluster_X': ['seq2', 'seq1']   # Different order
        }

        signatures = compute_all_signatures(current_clusters)

        is_equivalent = check_cluster_set_equivalence(
            input_cluster_ids=input_cluster_ids,
            output_clusters=output_clusters,
            current_clusters=current_clusters,
            cluster_signatures=signatures
        )

        assert is_equivalent, "Equivalence should be order-independent"


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

        cluster_signatures = compute_all_signatures(current_clusters)

        # Two non-overlapping operations that actually change the clusters
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {
                    'cluster_A_split1': ['seq1'],
                    'cluster_A_split2': ['seq2']
                },  # Split cluster A
                'scope_signature': frozenset(['cluster_A'])
            },
            {
                'seed_id': 'cluster_C',
                'input_cluster_ids': {'cluster_C'},
                'output_clusters': {
                    'cluster_C_split1': ['seq5'],
                    'cluster_C_split2': ['seq6']
                },  # Split cluster C
                'scope_signature': frozenset(['cluster_C'])
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations,
            cluster_signatures=cluster_signatures
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

        cluster_signatures = compute_all_signatures(current_clusters)

        # Two operations that try to modify cluster_A (both split it)
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {
                    'cluster_A_v1_1': ['seq1'],
                    'cluster_A_v1_2': ['seq2']
                },
                'scope_signature': frozenset(['cluster_A'])
            },
            {
                'seed_id': 'cluster_A',  # Same cluster (but already consumed by first op)
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {
                    'cluster_A_v2_1': ['seq1'],
                    'cluster_A_v2_2': ['seq2']
                },
                'scope_signature': frozenset(['cluster_A'])
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations,
            cluster_signatures=cluster_signatures
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

        cluster_signatures = compute_all_signatures(current_clusters)

        # Operation that produces identical output (converged)
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {'cluster_A_same': ['seq1', 'seq2']},  # Same content
                'scope_signature': frozenset(['cluster_A'])
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations,
            cluster_signatures=cluster_signatures
        )

        # Should detect convergence
        assert len(converged) == 1
        assert frozenset(['cluster_A']) in converged
        assert not changes_made, "No changes should be recorded for converged scope"

    def test_merge_operation_marks_changes(self):
        """Test that merge operations are marked as changes."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        cluster_signatures = compute_all_signatures(current_clusters)

        # Operation that merges two clusters
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A', 'cluster_B'},
                'output_clusters': {'cluster_merged': ['seq1', 'seq2', 'seq3', 'seq4']},
                'scope_signature': frozenset(['cluster_A', 'cluster_B'])
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations,
            cluster_signatures=cluster_signatures
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
        signatures = compute_all_signatures(clusters)

        assert signatures == {}
        assert check_full_set_equivalence(clusters, {})

    def test_single_cluster_input(self):
        """Test refinement with only one cluster."""
        clusters = {
            'cluster_1': ['seq1', 'seq2', 'seq3']
        }

        # Should compute signature correctly
        signatures = compute_all_signatures(clusters)
        assert len(signatures) == 1
        assert signatures['cluster_1'] == frozenset(['seq1', 'seq2', 'seq3'])

        # Should be equivalent to itself
        assert check_full_set_equivalence(clusters, clusters)

    def test_all_singleton_clusters(self):
        """Test refinement with all singleton clusters."""
        clusters = {
            f'cluster_{i}': [f'seq{i}']
            for i in range(100)
        }

        signatures = compute_all_signatures(clusters)
        assert len(signatures) == 100

        # Each signature should be a singleton frozenset
        for cluster_id, sig in signatures.items():
            assert len(sig) == 1

    def test_immediate_convergence_detected(self):
        """Test that immediate convergence (no changes) is detected."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        cluster_signatures = compute_all_signatures(current_clusters)

        # Operations that produce identical output (converged immediately)
        operations = [
            {
                'seed_id': 'cluster_A',
                'input_cluster_ids': {'cluster_A'},
                'output_clusters': {'cluster_A_same': ['seq1', 'seq2']},
                'scope_signature': frozenset(['cluster_A'])
            },
            {
                'seed_id': 'cluster_B',
                'input_cluster_ids': {'cluster_B'},
                'output_clusters': {'cluster_B_same': ['seq3', 'seq4']},
                'scope_signature': frozenset(['cluster_B'])
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations,
            cluster_signatures=cluster_signatures
        )

        # Both should be detected as converged
        assert len(converged) == 2
        assert not changes_made

    def test_very_large_cluster(self):
        """Test signature computation with a very large cluster."""
        # Simulate a cluster approaching max_full_gaphack_size (300)
        large_cluster_members = [f'seq{i:05d}' for i in range(250)]

        clusters = {
            'large_cluster': large_cluster_members,
            'small_cluster': ['seq_a', 'seq_b']
        }

        signatures = compute_all_signatures(clusters)

        assert len(signatures['large_cluster']) == 250
        assert len(signatures['small_cluster']) == 2

    def test_equivalence_with_empty_output(self):
        """Test that empty outputs are handled correctly."""
        input_cluster_ids = {'cluster_A'}
        current_clusters = {
            'cluster_A': ['seq1', 'seq2']
        }
        output_clusters = {}  # Empty output (all sequences filtered out?)

        signatures = compute_all_signatures(current_clusters)

        is_equivalent = check_cluster_set_equivalence(
            input_cluster_ids=input_cluster_ids,
            output_clusters=output_clusters,
            current_clusters=current_clusters,
            cluster_signatures=signatures
        )

        # Empty output is not equivalent to non-empty input
        assert not is_equivalent

    def test_operations_with_no_input_clusters(self):
        """Test operations that reference non-existent clusters are skipped."""
        current_clusters = {
            'cluster_A': ['seq1', 'seq2'],
            'cluster_B': ['seq3', 'seq4']
        }

        cluster_signatures = compute_all_signatures(current_clusters)

        # Operation referencing cluster that doesn't exist
        operations = [
            {
                'seed_id': 'cluster_NONEXISTENT',
                'input_cluster_ids': {'cluster_NONEXISTENT'},
                'output_clusters': {'cluster_new': ['seq5', 'seq6']},
                'scope_signature': frozenset(['cluster_NONEXISTENT'])
            }
        ]

        next_clusters, changes_made, converged = execute_refinement_operations(
            current_clusters=current_clusters,
            operations=operations,
            cluster_signatures=cluster_signatures
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

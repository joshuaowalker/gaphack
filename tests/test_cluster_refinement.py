"""
Tests for cluster refinement algorithms.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict
from hypothesis import given, strategies as st, settings
import numpy as np

from gaphack.cluster_refinement import (
    RefinementConfig,
    apply_full_gaphack_to_scope,
    apply_full_gaphack_to_scope_with_metadata,
    verify_no_conflicts
)


class TestRefinementConfig:
    """Test suite for RefinementConfig class."""

    def test_default_initialization(self):
        """Test RefinementConfig with default parameters."""
        config = RefinementConfig()

        assert config.max_full_gaphack_size == 300
        assert config.close_threshold is None
        assert config.max_iterations == 10
        assert config.k_neighbors == 20
        assert config.search_method == "blast"

    def test_custom_initialization(self):
        """Test RefinementConfig with custom parameters."""
        config = RefinementConfig(
            max_full_gaphack_size=500,
            close_threshold=0.025,
            max_iterations=15,
            k_neighbors=30,
            search_method="vsearch"
        )

        assert config.max_full_gaphack_size == 500
        assert config.close_threshold == 0.025
        assert config.max_iterations == 15
        assert config.k_neighbors == 30
        assert config.search_method == "vsearch"


class TestFullGapHACkApplication:
    """Test suite for full gapHACk application to scopes."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = ["ATCGATCG", "ATCGATCC", "TTTTGGGG", "TTTTGGCC"]
        self.test_headers = ["seq_0", "seq_1", "seq_2", "seq_3"]

    @patch('gaphack.distance_providers.MSACachedDistanceProvider')
    @patch('gaphack.cluster_refinement.GapOptimizedClustering')
    def test_apply_full_gaphack_to_scope_basic(self, mock_clustering_class, mock_msa_provider):
        """Test basic full gapHACk application to scope."""
        # Mock MSA distance provider
        mock_provider = Mock()
        mock_provider.build_distance_matrix.return_value = np.array([[0.0, 0.1], [0.1, 0.0]])
        mock_msa_provider.return_value = mock_provider

        # Mock clustering result
        mock_clusterer = Mock()
        mock_clusterer.cluster.return_value = ([[0, 1]], [], {})
        mock_clustering_class.return_value = mock_clusterer

        # Mock distance provider (not used by refinement but passed in)
        mock_distance_provider = Mock()

        result = apply_full_gaphack_to_scope(["ATCGATCG", "ATCGATCC"], ["seq_0", "seq_1"])

        assert len(result) == 1  # One cluster
        cluster_values = list(result.values())
        assert "seq_0" in cluster_values[0]
        assert "seq_1" in cluster_values[0]

    @patch('gaphack.distance_providers.MSACachedDistanceProvider')
    @patch('gaphack.cluster_refinement.GapOptimizedClustering')
    def test_apply_full_gaphack_to_scope_with_metadata(self, mock_clustering_class, mock_msa_provider):
        """Test full gapHACk application with metadata return."""
        # Mock MSA distance provider
        mock_provider = Mock()
        mock_provider.build_distance_matrix.return_value = np.array([[0.0, 0.1], [0.1, 0.0]])
        mock_msa_provider.return_value = mock_provider

        # Mock clustering result with metadata
        mock_clusterer = Mock()
        mock_clusterer.cluster.return_value = ([[0], [1]], [], {'best_config': {'gap_size': 0.05}})
        mock_clustering_class.return_value = mock_clusterer

        mock_distance_provider = Mock()

        clusters, metadata = apply_full_gaphack_to_scope_with_metadata(["ATCGATCG", "ATCGATCC"], ["seq_0", "seq_1"])

        assert len(clusters) == 2  # Two clusters
        assert metadata['gap_size'] == 0.05


class TestConflictVerification:
    """Test suite for conflict verification algorithms."""

    def test_verify_no_conflicts_clean(self):
        """Test conflict verification with clean clusters."""
        clusters = {
            "cluster_1": ["seq_1", "seq_2"],
            "cluster_2": ["seq_3", "seq_4"],
            "cluster_3": ["seq_5"]
        }

        result = verify_no_conflicts(clusters)

        assert result['no_conflicts'] is True
        assert result['conflict_count'] == 0
        assert result['total_sequences'] == 5
        assert result['total_assignments'] == 5

    def test_verify_no_conflicts_with_conflicts(self):
        """Test conflict verification with conflicts present."""
        clusters = {
            "cluster_1": ["seq_1", "seq_2"],
            "cluster_2": ["seq_2", "seq_3"],  # seq_2 conflicts
            "cluster_3": ["seq_4"]
        }

        result = verify_no_conflicts(clusters)

        assert result['no_conflicts'] is False
        assert result['conflict_count'] == 1
        assert 'seq_2' in result['conflicts']
        assert result['conflicts']['seq_2'] == ["cluster_1", "cluster_2"]

    def test_verify_no_conflicts_with_original_comparison(self):
        """Test conflict verification with original conflict comparison."""
        clusters = {
            "cluster_1": ["seq_1", "seq_2"],
            "cluster_2": ["seq_3"],
            "cluster_3": ["seq_4", "seq_5"],
            "cluster_4": ["seq_5"]  # New conflict
        }

        original_conflicts = {"seq_2": ["cluster_A", "cluster_B"]}

        result = verify_no_conflicts(clusters, original_conflicts)

        assert result['no_conflicts'] is False
        assert len(result['resolved_conflicts']) == 1  # seq_2 was resolved
        assert 'seq_2' in result['resolved_conflicts']
        assert len(result['new_conflicts']) == 1  # seq_5 is new conflict
        assert 'seq_5' in result['new_conflicts']

    def test_verify_no_conflicts_final_context(self):
        """Test conflict verification with final context."""
        clusters = {
            "cluster_1": ["seq_1", "seq_2"],
            "cluster_2": ["seq_2", "seq_3"]  # Conflict
        }

        result = verify_no_conflicts(clusters, context="final")

        assert result['no_conflicts'] is False
        assert result['critical_failure'] is True
        assert result['verification_context'] == "final"


class TestConvergenceMetrics:
    """Test suite for convergence metrics computation."""

    @patch('gaphack.distance_providers.MSACachedDistanceProvider')
    def test_compute_convergence_metrics_basic(self, mock_msa_provider_class):
        """Test basic convergence metrics computation."""
        from gaphack.cluster_refinement import compute_convergence_metrics, _convergence_metrics_cache

        # Clear cache to avoid interference from other tests
        _convergence_metrics_cache.clear()

        # Create mock distance provider that returns different distances
        # Set up distance matrix: cluster1 has internal distance 0.01, cluster2 has 0.02
        # Inter-cluster distances are 0.1
        def create_mock_provider(*args, **kwargs):
            mock_provider = Mock()
            def get_distance(i, j):
                if i == j:
                    return 0.0
                # Simple distance based on scope index positions
                elif abs(i - j) == 1:  # Adjacent sequences (within cluster)
                    return 0.01
                else:  # Between clusters
                    return 0.1
            mock_provider.get_distance = get_distance
            return mock_provider

        mock_msa_provider_class.side_effect = create_mock_provider

        # Create test clusters
        clusters = {
            'cluster_1': ['seq_0', 'seq_1'],
            'cluster_2': ['seq_2', 'seq_3']
        }
        headers = ['seq_0', 'seq_1', 'seq_2', 'seq_3']
        sequences = ['ATCG', 'ATCC', 'GGTA', 'GGTC']

        # Mock proximity graph to return neighbors
        mock_graph = Mock()
        mock_graph.get_k_nearest_neighbors = Mock()
        mock_graph.get_k_nearest_neighbors.side_effect = lambda cluster_id, k: [
            ('cluster_2', 0.1) if cluster_id == 'cluster_1' else ('cluster_1', 0.1)
        ]

        # Compute metrics
        metrics = compute_convergence_metrics(
            clusters=clusters,
            proximity_graph=mock_graph,
            sequences=sequences,
            headers=headers,
            target_percentile=95,
            show_progress=False
        )

        # Verify all four metrics are present
        assert 'mean_gap' in metrics
        assert 'weighted_gap' in metrics
        assert 'gap_coverage' in metrics
        assert 'gap_coverage_sequences' in metrics

        # Verify gap is positive (inter-cluster distance > intra-cluster distance)
        assert metrics['mean_gap'] > 0
        assert metrics['weighted_gap'] > 0

        # All clusters should have positive gap
        assert metrics['gap_coverage'] == 1.0
        assert metrics['gap_coverage_sequences'] == 1.0

    @patch('gaphack.distance_providers.MSACachedDistanceProvider')
    def test_compute_convergence_metrics_singleton(self, mock_msa_provider_class):
        """Test convergence metrics with singleton clusters."""
        from gaphack.cluster_refinement import compute_convergence_metrics, _convergence_metrics_cache

        # Clear cache to avoid interference from other tests
        _convergence_metrics_cache.clear()

        # Create mock distance provider
        def create_mock_provider(*args, **kwargs):
            mock_provider = Mock()
            def get_distance(i, j):
                if i == j:
                    return 0.0
                else:
                    return 0.1  # All inter-cluster distances
            mock_provider.get_distance = get_distance
            return mock_provider

        mock_msa_provider_class.side_effect = create_mock_provider

        # Test with singletons
        clusters = {
            'cluster_1': ['seq_0'],
            'cluster_2': ['seq_1']
        }
        headers = ['seq_0', 'seq_1']
        sequences = ['ATCG', 'GGTA']

        # Mock proximity graph
        mock_graph = Mock()
        mock_graph.get_k_nearest_neighbors = Mock()
        mock_graph.get_k_nearest_neighbors.return_value = [('cluster_2', 0.1)]

        metrics = compute_convergence_metrics(
            clusters=clusters,
            proximity_graph=mock_graph,
            sequences=sequences,
            headers=headers,
            target_percentile=95,
            show_progress=False
        )

        # Singletons should have intra=0, so gap = inter_lower - 0
        assert metrics['mean_gap'] > 0
        assert metrics['gap_coverage'] > 0

    @patch('gaphack.distance_providers.MSACachedDistanceProvider')
    def test_compute_convergence_metrics_negative_gap(self, mock_msa_provider_class):
        """Test convergence metrics when some clusters have negative gaps."""
        from gaphack.cluster_refinement import compute_convergence_metrics, _convergence_metrics_cache

        # Clear cache to avoid interference from other tests
        _convergence_metrics_cache.clear()

        # Create mock distance provider
        # Cluster1 has large internal distances (0.2), cluster2 small (0.01)
        # Inter-cluster is 0.05 - cluster1 will have negative gap
        def create_mock_provider(scope_sequences, scope_headers):
            mock_provider = Mock()
            def get_distance(i, j):
                if i == j:
                    return 0.0
                # Determine which cluster each index belongs to based on scope
                header_i = scope_headers[i]
                header_j = scope_headers[j]

                # cluster_1 is seq_0, seq_1 (high intra-cluster distance)
                # cluster_2 is seq_2, seq_3 (low intra-cluster distance)
                cluster1_seqs = {'seq_0', 'seq_1'}
                cluster2_seqs = {'seq_2', 'seq_3'}

                in_cluster1_i = header_i in cluster1_seqs
                in_cluster1_j = header_j in cluster1_seqs
                in_cluster2_i = header_i in cluster2_seqs
                in_cluster2_j = header_j in cluster2_seqs

                if in_cluster1_i and in_cluster1_j:
                    return 0.2  # High intra-cluster1
                elif in_cluster2_i and in_cluster2_j:
                    return 0.01  # Low intra-cluster2
                else:
                    return 0.05  # Inter-cluster

            mock_provider.get_distance = get_distance
            return mock_provider

        mock_msa_provider_class.side_effect = create_mock_provider

        clusters = {
            'cluster_1': ['seq_0', 'seq_1'],
            'cluster_2': ['seq_2', 'seq_3']
        }
        headers = ['seq_0', 'seq_1', 'seq_2', 'seq_3']
        sequences = ['ATCG', 'ATCC', 'GGTA', 'GGTC']

        # Mock proximity graph
        mock_graph = Mock()
        mock_graph.get_k_nearest_neighbors = Mock()
        mock_graph.get_k_nearest_neighbors.side_effect = lambda cluster_id, k: [
            ('cluster_2', 0.05) if cluster_id == 'cluster_1' else ('cluster_1', 0.05)
        ]

        metrics = compute_convergence_metrics(
            clusters=clusters,
            proximity_graph=mock_graph,
            sequences=sequences,
            headers=headers,
            target_percentile=95,
            show_progress=False
        )

        # Gap coverage should be < 1.0 because cluster1 has negative gap
        assert metrics['gap_coverage'] < 1.0
        assert metrics['gap_coverage_sequences'] < 1.0


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_verify_conflicts_empty_clusters(self):
        """Test conflict verification with empty cluster dictionary."""
        result = verify_no_conflicts({})

        assert result['no_conflicts'] is True
        assert result['conflict_count'] == 0
        assert result['total_sequences'] == 0
        assert result['total_assignments'] == 0
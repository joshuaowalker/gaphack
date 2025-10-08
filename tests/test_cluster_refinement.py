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
    find_conflict_components,
    apply_full_gaphack_to_scope,
    apply_full_gaphack_to_scope_with_metadata,
    resolve_conflicts,
    verify_no_conflicts
)


class TestRefinementConfig:
    """Test suite for RefinementConfig class."""

    def test_default_initialization(self):
        """Test RefinementConfig with default parameters."""
        config = RefinementConfig()

        assert config.max_full_gaphack_size == 300
        assert config.context_threshold_multiplier == 2.0
        assert config.close_threshold is None
        assert config.max_iterations == 10
        assert config.k_neighbors == 20
        assert config.search_method == "blast"

    def test_custom_initialization(self):
        """Test RefinementConfig with custom parameters."""
        config = RefinementConfig(
            max_full_gaphack_size=500,
            context_threshold_multiplier=3.0,
            close_threshold=0.025,
            max_iterations=15,
            k_neighbors=30,
            search_method="vsearch"
        )

        assert config.max_full_gaphack_size == 500
        assert config.context_threshold_multiplier == 3.0
        assert config.close_threshold == 0.025
        assert config.max_iterations == 15
        assert config.k_neighbors == 30
        assert config.search_method == "vsearch"


class TestConflictComponents:
    """Test suite for conflict component finding algorithms."""

    def test_find_conflict_components_simple(self):
        """Test conflict component finding with simple conflicts."""
        conflicts = {
            "seq_1": ["cluster_A", "cluster_B"],
            "seq_2": ["cluster_B", "cluster_C"]
        }
        all_clusters = {
            "cluster_A": ["seq_1", "seq_3"],
            "cluster_B": ["seq_1", "seq_2", "seq_4"],
            "cluster_C": ["seq_2", "seq_5"]
        }

        components = find_conflict_components(conflicts, all_clusters)

        # Should find one connected component containing all three clusters
        assert len(components) == 1
        assert set(components[0]) == {"cluster_A", "cluster_B", "cluster_C"}

    def test_find_conflict_components_isolated(self):
        """Test conflict component finding with isolated conflicts."""
        conflicts = {
            "seq_1": ["cluster_A", "cluster_B"],
            "seq_2": ["cluster_C", "cluster_D"]
        }
        all_clusters = {
            "cluster_A": ["seq_1"],
            "cluster_B": ["seq_1"],
            "cluster_C": ["seq_2"],
            "cluster_D": ["seq_2"]
        }

        components = find_conflict_components(conflicts, all_clusters)

        # Should find two separate components
        assert len(components) == 2
        component_sets = [set(comp) for comp in components]
        assert {"cluster_A", "cluster_B"} in component_sets
        assert {"cluster_C", "cluster_D"} in component_sets

    def test_find_conflict_components_empty(self):
        """Test conflict component finding with no conflicts."""
        conflicts = {}
        all_clusters = {"cluster_A": ["seq_1"], "cluster_B": ["seq_2"]}

        components = find_conflict_components(conflicts, all_clusters)

        assert components == []


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


class TestConflictResolution:
    """Test suite for conflict resolution algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = ["ATCGATCG", "ATCGATCC", "TTTTGGGG", "TTTTGGCC"]
        self.test_headers = ["seq_0", "seq_1", "seq_2", "seq_3"]
        self.test_clusters = {
            "cluster_A": ["seq_0", "seq_1"],
            "cluster_B": ["seq_1", "seq_2"],  # seq_1 conflicts
            "cluster_C": ["seq_3"]
        }
        self.conflicts = {"seq_1": ["cluster_A", "cluster_B"]}

    @patch('gaphack.cluster_refinement.apply_full_gaphack_to_scope')
    def test_resolve_conflicts_simple(self, mock_full_gaphack):
        """Test simple conflict resolution."""
        # Mock full gapHACk result
        mock_full_gaphack.return_value = {
            "classic_1": ["seq_0", "seq_1"],
            "classic_2": ["seq_2"]
        }

        updated_clusters, tracking_info = resolve_conflicts(
            self.conflicts, self.test_clusters, self.test_sequences,
            self.test_headers
        )

        # Should remove conflicted clusters and add classic results
        assert "cluster_A" not in updated_clusters
        assert "cluster_B" not in updated_clusters
        assert "cluster_C" in updated_clusters  # Non-conflicted cluster preserved
        assert "classic_1" in updated_clusters
        assert "classic_2" in updated_clusters

        # Verify tracking info
        assert tracking_info.stage_name == "Conflict Resolution"
        assert tracking_info.summary_stats['conflicts_count'] == 1

    def test_resolve_conflicts_empty(self):
        """Test conflict resolution with no conflicts."""
        updated_clusters, tracking_info = resolve_conflicts(
            {}, self.test_clusters, self.test_sequences,
            self.test_headers
        )

        # Should return unchanged clusters
        assert updated_clusters == self.test_clusters
        assert tracking_info.summary_stats['conflicts_count'] == 0

    @patch('gaphack.cluster_refinement.apply_full_gaphack_to_scope')
    def test_resolve_conflicts_oversized(self, mock_full_gaphack):
        """Test conflict resolution with oversized component."""
        # Create large conflict scope
        large_clusters = {}
        conflicts = {}
        for i in range(100):  # Create 100 clusters with 10 sequences each
            cluster_id = f"cluster_{i}"
            sequence_ids = [f"seq_{i*10 + j}" for j in range(10)]
            large_clusters[cluster_id] = sequence_ids

            # Create conflict for first sequence
            if i < 50:  # First 50 clusters conflict on seq_0
                if "seq_0" not in conflicts:
                    conflicts["seq_0"] = []
                conflicts["seq_0"].append(cluster_id)

        config = RefinementConfig(max_full_gaphack_size=100)  # Small size limit
        updated_clusters, tracking_info = resolve_conflicts(
            conflicts, large_clusters, [], [],
            config
        )

        # Should skip oversized component
        assert len(tracking_info.components_processed) == 1
        component_info = tracking_info.components_processed[0]
        assert component_info['processed'] is False
        assert component_info['skipped_reason'] == 'oversized'


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


class TestPropertyBasedRefinement:
    """Property-based tests for refinement algorithms."""

    @given(
        conflicts=st.dictionaries(
            st.text(alphabet="seq_", min_size=4, max_size=10),
            st.lists(st.text(alphabet="cluster_", min_size=8, max_size=15), min_size=2, max_size=4),
            min_size=1, max_size=5
        )
    )
    @settings(deadline=500)
    def test_find_conflict_components_properties(self, conflicts):
        """Property-based test for conflict component finding."""
        # Generate corresponding cluster data
        all_clusters = {}
        for seq_id, cluster_ids in conflicts.items():
            for cluster_id in cluster_ids:
                if cluster_id not in all_clusters:
                    all_clusters[cluster_id] = []
                if seq_id not in all_clusters[cluster_id]:
                    all_clusters[cluster_id].append(seq_id)

        components = find_conflict_components(conflicts, all_clusters)

        # Properties that should always hold
        # 1. All conflicted clusters should appear in exactly one component
        all_conflict_clusters = set()
        for cluster_ids in conflicts.values():
            all_conflict_clusters.update(cluster_ids)

        found_clusters = set()
        for component in components:
            found_clusters.update(component)

        assert found_clusters == all_conflict_clusters

        # 2. No cluster should appear in multiple components
        cluster_counts = defaultdict(int)
        for component in components:
            for cluster_id in component:
                cluster_counts[cluster_id] += 1

        for count in cluster_counts.values():
            assert count == 1


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_verify_conflicts_empty_clusters(self):
        """Test conflict verification with empty cluster dictionary."""
        result = verify_no_conflicts({})

        assert result['no_conflicts'] is True
        assert result['conflict_count'] == 0
        assert result['total_sequences'] == 0
        assert result['total_assignments'] == 0
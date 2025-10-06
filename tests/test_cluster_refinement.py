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
    ExpandedScope,
    find_conflict_components,
    apply_full_gaphack_to_scope,
    apply_full_gaphack_to_scope_with_metadata,
    resolve_conflicts,
    find_connected_close_components,
    needs_minimal_context_for_gap_calculation,
    add_context_at_distance_threshold,
    expand_context_for_gap_optimization,
    refine_close_clusters,
    verify_no_conflicts,
    MAX_FULL_GAPHACK_SIZE,
    PREFERRED_SCOPE_SIZE,
    EXPANSION_SIZE_BUFFER
)


class TestRefinementConfig:
    """Test suite for RefinementConfig class."""

    def test_default_initialization(self):
        """Test RefinementConfig with default parameters."""
        config = RefinementConfig()

        assert config.max_full_gaphack_size == 300
        assert config.preferred_scope_size == 250
        assert config.expansion_size_buffer == 50
        assert config.conflict_expansion_threshold is None
        assert config.close_cluster_expansion_threshold is None
        assert config.incremental_search_distance is None
        assert config.max_closest_clusters == 5

    def test_custom_initialization(self):
        """Test RefinementConfig with custom parameters."""
        config = RefinementConfig(
            max_full_gaphack_size=500,
            preferred_scope_size=400,
            expansion_size_buffer=100,
            conflict_expansion_threshold=0.03,
            close_cluster_expansion_threshold=0.025,
            incremental_search_distance=0.05,
            max_closest_clusters=10
        )

        assert config.max_full_gaphack_size == 500
        assert config.preferred_scope_size == 400
        assert config.expansion_size_buffer == 100
        assert config.conflict_expansion_threshold == 0.03
        assert config.close_cluster_expansion_threshold == 0.025
        assert config.incremental_search_distance == 0.05
        assert config.max_closest_clusters == 10


class TestExpandedScope:
    """Test suite for ExpandedScope class."""

    def test_expanded_scope_creation(self):
        """Test ExpandedScope creation and attributes."""
        sequences = ["ATCG", "GCTA", "CGAT"]
        headers = ["seq_0", "seq_1", "seq_2"]
        cluster_ids = ["cluster_1", "cluster_2"]

        scope = ExpandedScope(sequences, headers, cluster_ids)

        assert scope.sequences == sequences
        assert scope.headers == headers
        assert scope.cluster_ids == cluster_ids


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

    def test_find_connected_close_components_linear(self):
        """Test close component finding with linear connections."""
        close_pairs = [
            ("cluster_A", "cluster_B", 0.01),
            ("cluster_B", "cluster_C", 0.015),
            ("cluster_D", "cluster_E", 0.012)
        ]

        components = find_connected_close_components(close_pairs)

        assert len(components) == 2
        component_sets = [set(comp) for comp in components]
        assert {"cluster_A", "cluster_B", "cluster_C"} in component_sets
        assert {"cluster_D", "cluster_E"} in component_sets

    def test_find_connected_close_components_empty(self):
        """Test close component finding with no close pairs."""
        close_pairs = []

        components = find_connected_close_components(close_pairs)

        assert components == []


class TestScopeExpansion:
    """Test suite for scope expansion algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_clusters = {
            "cluster_1": ["seq_1", "seq_2"],
            "cluster_2": ["seq_3", "seq_4"],
            "cluster_3": ["seq_5", "seq_6"],
            "cluster_4": ["seq_7", "seq_8"]
        }

    def test_needs_minimal_context_single_cluster(self):
        """Test minimal context detection with single cluster."""
        mock_graph = Mock()

        result = needs_minimal_context_for_gap_calculation(
            ["cluster_1"], self.test_clusters, mock_graph, max_lump=0.02
        )

        assert result is True  # Single cluster always needs context

    def test_needs_minimal_context_distant_clusters(self):
        """Test minimal context detection with distant clusters."""
        mock_graph = Mock()
        mock_graph.get_neighbors_within_distance.return_value = []  # No neighbors within 2*max_lump

        result = needs_minimal_context_for_gap_calculation(
            ["cluster_1", "cluster_2"], self.test_clusters, mock_graph, max_lump=0.02
        )

        assert result is False  # Distant clusters don't need context

    def test_needs_minimal_context_close_clusters(self):
        """Test minimal context detection with close clusters."""
        mock_graph = Mock()
        mock_graph.get_neighbors_within_distance.return_value = [("cluster_2", 0.01)]

        result = needs_minimal_context_for_gap_calculation(
            ["cluster_1", "cluster_2"], self.test_clusters, mock_graph, max_lump=0.02
        )

        assert result is True  # Close clusters need context

    def test_add_context_at_distance_threshold_success(self):
        """Test successful context addition at distance threshold."""
        current_cluster_ids = {"cluster_1"}

        mock_graph = Mock()
        mock_graph.get_k_nearest_neighbors.return_value = [
            ("cluster_2", 0.025),  # Beyond threshold
            ("cluster_3", 0.035)   # Further beyond threshold
        ]

        success, cluster_id, distance = add_context_at_distance_threshold(
            current_cluster_ids, self.test_clusters, mock_graph,
            distance_threshold=0.02, max_scope_size=10
        )

        assert success is True
        assert cluster_id == "cluster_2"
        assert distance == 0.025

    def test_add_context_at_distance_threshold_size_limit(self):
        """Test context addition failure due to size limit."""
        current_cluster_ids = {"cluster_1"}

        mock_graph = Mock()
        mock_graph.get_k_nearest_neighbors.return_value = [("cluster_2", 0.025)]

        success, cluster_id, distance = add_context_at_distance_threshold(
            current_cluster_ids, self.test_clusters, mock_graph,
            distance_threshold=0.02, max_scope_size=3  # Too small for cluster_2
        )

        assert success is False
        assert cluster_id == ""
        assert distance == 0.0


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


class TestCloseClusterRefinement:
    """Test suite for close cluster refinement algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_clusters = {
            "cluster_1": ["seq_1", "seq_2"],
            "cluster_2": ["seq_3", "seq_4"],
            "cluster_3": ["seq_5", "seq_6"]
        }
        self.test_sequences = ["ATCGATCG", "ATCGATCC", "TTTTGGGG", "TTTTGGCC", "GGGGCCCC", "GGGGCCCT"]
        self.test_headers = ["seq_1", "seq_2", "seq_3", "seq_4", "seq_5", "seq_6"]

    @patch('gaphack.cluster_refinement.expand_context_for_gap_optimization')
    def test_refine_close_clusters_basic(self, mock_expand_context):
        """Test basic close cluster refinement."""
        # Mock proximity graph
        mock_graph = Mock()
        mock_graph.find_close_pairs.return_value = [("cluster_1", "cluster_2", 0.015)]

        # Mock expansion result
        mock_scope = ExpandedScope(
            sequences=["seq_1", "seq_2", "seq_3", "seq_4"],
            headers=["seq_1", "seq_2", "seq_3", "seq_4"],
            cluster_ids=["cluster_1", "cluster_2"]
        )
        mock_result = {
            "refined_1": ["seq_1", "seq_2", "seq_3"],
            "refined_2": ["seq_4"]
        }
        mock_expand_context.return_value = (mock_scope, mock_result)

        updated_clusters, tracking_info = refine_close_clusters(
            self.test_clusters, self.test_sequences, self.test_headers,
            mock_graph, close_threshold=0.02
        )

        # Should replace close clusters with refined result
        assert "cluster_1" not in updated_clusters
        assert "cluster_2" not in updated_clusters
        assert "cluster_3" in updated_clusters  # Not close, preserved
        assert "refined_1" in updated_clusters
        assert "refined_2" in updated_clusters

        # Verify tracking info
        assert tracking_info.stage_name == "Close Cluster Refinement"
        assert tracking_info.summary_stats['close_pairs_found'] == 1

    def test_refine_close_clusters_no_close_pairs(self):
        """Test close cluster refinement with no close pairs."""
        mock_graph = Mock()
        mock_graph.find_close_pairs.return_value = []

        updated_clusters, tracking_info = refine_close_clusters(
            self.test_clusters, self.test_sequences, self.test_headers,
            mock_graph
        )

        # Should return unchanged clusters
        assert updated_clusters == self.test_clusters
        assert tracking_info.summary_stats['close_pairs_found'] == 0

    def test_refine_close_clusters_insufficient_clusters(self):
        """Test close cluster refinement with insufficient clusters."""
        single_cluster = {"cluster_1": ["seq_1", "seq_2"]}

        mock_graph = Mock()
        updated_clusters, tracking_info = refine_close_clusters(
            single_cluster, self.test_sequences, self.test_headers,
            mock_graph
        )

        assert updated_clusters == single_cluster
        assert tracking_info.summary_stats['components_processed_count'] == 0


class TestContextExpansion:
    """Test suite for context expansion algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_clusters = {
            "cluster_1": ["seq_1", "seq_2"],
            "cluster_2": ["seq_3", "seq_4"],
            "cluster_3": ["seq_5", "seq_6"]
        }
        # Provide actual sequences and headers for mapping
        self.test_headers = ["seq_1", "seq_2", "seq_3", "seq_4", "seq_5", "seq_6"]
        self.test_sequences = ["ACGT", "ACGG", "TTTT", "TTTG", "GGGG", "GGGC"]

    @patch('gaphack.cluster_refinement.apply_full_gaphack_to_scope_with_metadata')
    def test_expand_context_for_gap_optimization_success(self, mock_full_gaphack):
        """Test successful context expansion for gap optimization."""
        # Mock distance provider
        mock_distance_provider = Mock()

        # Mock proximity graph
        mock_graph = Mock()
        # First iteration context distance = max_lump * 1.5 = 0.02 * 1.5 = 0.03
        # Second iteration context distance = max_lump * 2.0 = 0.04
        mock_graph.get_k_nearest_neighbors.return_value = [
            ("cluster_2", 0.025),  # Too close for first iteration (< 0.03)
            ("cluster_3", 0.035)   # Suitable for first iteration (> 0.03)
        ]

        # Mock full gapHACk results - first call fails, second succeeds
        mock_full_gaphack.side_effect = [
            ({}, {'gap_size': -0.001}),  # First iteration: negative gap
            ({"result_1": ["seq_1", "seq_2"]}, {'gap_size': 0.005})  # Second: positive gap
        ]

        core_cluster_ids = ["cluster_1"]

        scope, result = expand_context_for_gap_optimization(
            core_cluster_ids, self.test_clusters, self.test_sequences, self.test_headers,
            mock_graph,
            expansion_threshold=0.03, max_scope_size=10,
            max_lump=0.02, min_split=0.005, target_percentile=95,
            target_gap=0.001, max_iterations=3
        )

        assert result == {"result_1": ["seq_1", "seq_2"]}
        assert "cluster_3" in scope.cluster_ids  # Context was added (beyond 0.03 threshold)

    @patch('gaphack.cluster_refinement.apply_full_gaphack_to_scope_with_metadata')
    def test_expand_context_for_gap_optimization_max_iterations(self, mock_full_gaphack):
        """Test context expansion hitting max iterations."""
        mock_distance_provider = Mock()

        mock_graph = Mock()
        mock_graph.get_k_nearest_neighbors.return_value = [
            ("cluster_2", 0.025),
            ("cluster_3", 0.035)
        ]

        # Mock full gapHACk to always return negative gap
        mock_full_gaphack.return_value = ({"result_1": ["seq_1"]}, {'gap_size': -0.001})

        core_cluster_ids = ["cluster_1"]

        scope, result = expand_context_for_gap_optimization(
            core_cluster_ids, self.test_clusters, self.test_sequences, self.test_headers,
            mock_graph,
            expansion_threshold=0.03, max_scope_size=10,
            max_lump=0.02, min_split=0.005, target_percentile=95,
            target_gap=0.001, max_iterations=2
        )

        # Should return best result even if target not achieved
        assert result == {"result_1": ["seq_1"]}


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


class TestConstants:
    """Test suite for module constants."""

    def test_module_constants(self):
        """Test that module constants are properly defined."""
        assert MAX_FULL_GAPHACK_SIZE == 300
        assert PREFERRED_SCOPE_SIZE == 250
        assert EXPANSION_SIZE_BUFFER == 50


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

    @given(
        close_pairs=st.lists(
            st.tuples(
                st.text(alphabet="cluster_", min_size=8, max_size=15),
                st.text(alphabet="cluster_", min_size=8, max_size=15),
                st.floats(min_value=0.001, max_value=0.1)
            ).filter(lambda x: x[0] != x[1]),  # Ensure different cluster IDs
            min_size=1, max_size=10
        )
    )
    @settings(deadline=500)
    def test_find_connected_close_components_properties(self, close_pairs):
        """Property-based test for close component finding."""
        components = find_connected_close_components(close_pairs)

        # Property: All clusters in close pairs should appear in exactly one component
        all_clusters = set()
        for cluster1, cluster2, _ in close_pairs:
            all_clusters.add(cluster1)
            all_clusters.add(cluster2)

        found_clusters = set()
        for component in components:
            found_clusters.update(component)

        assert found_clusters == all_clusters

        # Property: No cluster appears in multiple components
        cluster_counts = defaultdict(int)
        for component in components:
            for cluster_id in component:
                cluster_counts[cluster_id] += 1

        for count in cluster_counts.values():
            assert count == 1


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""

    def test_context_expansion_no_available_clusters(self):
        """Test context expansion when no clusters are available."""
        current_cluster_ids = {"cluster_1"}
        all_clusters = {"cluster_1": ["seq_1"]}

        mock_graph = Mock()
        mock_graph.get_k_nearest_neighbors.return_value = []  # No neighbors

        success, cluster_id, distance = add_context_at_distance_threshold(
            current_cluster_ids, all_clusters, mock_graph,
            distance_threshold=0.02, max_scope_size=10
        )

        assert success is False
        assert cluster_id == ""
        assert distance == 0.0

    def test_verify_conflicts_empty_clusters(self):
        """Test conflict verification with empty cluster dictionary."""
        result = verify_no_conflicts({})

        assert result['no_conflicts'] is True
        assert result['conflict_count'] == 0
        assert result['total_sequences'] == 0
        assert result['total_assignments'] == 0
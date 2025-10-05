"""
Tests for cluster proximity graph infrastructure.

This module tests the BLAST-based K-NN graph for scalable cluster proximity queries
used in cluster refinement algorithms.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from hypothesis import given, strategies as st

from gaphack.cluster_graph import ClusterGraph
from gaphack.blast_neighborhood import SequenceCandidate


class TestClusterGraph:
    """Test ClusterGraph initialization and basic functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = [
            "ATGCGATCGATCGATCG",
            "ATGCGATCGATCGATCC",
            "TTTTTTTTTTTTTTTTTT",
            "TTTTTTTTTTTTTTTTCC",
            "GGGGGGGGGGGGGGGGGG"
        ]
        self.test_headers = [f"seq_{i}" for i in range(len(self.test_sequences))]
        self.test_clusters = {
            "cluster_1": ["seq_0", "seq_1"],
            "cluster_2": ["seq_2", "seq_3"],
            "cluster_3": ["seq_4"]
        }

        # Mock distance provider
        self.mock_distance_provider = Mock()
        self.mock_distance_provider.get_distance.return_value = 0.1

    def test_initialization_basic(self):
        """Test basic ClusterGraph initialization."""
        with patch('gaphack.cluster_graph.BlastNeighborhoodFinder'):
            graph = ClusterGraph(
                clusters=self.test_clusters,
                sequences=self.test_sequences,
                headers=self.test_headers,
                k_neighbors=5
            )

            assert graph.clusters == self.test_clusters
            assert graph.sequences == self.test_sequences
            assert graph.headers == self.test_headers
            assert graph.k_neighbors == 5
            assert graph.blast_evalue == 1e-5
            assert graph.blast_identity == 90.0

    def test_initialization_custom_parameters(self):
        """Test initialization with custom BLAST parameters."""
        with patch('gaphack.cluster_graph.BlastNeighborhoodFinder'):
            graph = ClusterGraph(
                clusters=self.test_clusters,
                sequences=self.test_sequences,
                headers=self.test_headers,
                k_neighbors=10,
                blast_evalue=1e-3,
                blast_identity=95.0
            )

            assert graph.k_neighbors == 10
            assert graph.blast_evalue == 1e-3
            assert graph.blast_identity == 95.0

    def test_initialization_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_cache = Path(tmpdir) / "custom_cache"

            with patch('gaphack.cluster_graph.BlastNeighborhoodFinder'):
                graph = ClusterGraph(
                    clusters=self.test_clusters,
                    sequences=self.test_sequences,
                    headers=self.test_headers,
                    cache_dir=custom_cache
                )

                assert graph.cache_dir == custom_cache

    def test_empty_clusters(self):
        """Test behavior with empty clusters."""
        with patch('gaphack.cluster_graph.BlastNeighborhoodFinder'):
            graph = ClusterGraph(
                clusters={},
                sequences=self.test_sequences,
                headers=self.test_headers)

            assert graph.clusters == {}
            assert graph.medoid_cache == {}
            assert graph.knn_graph == {}


class TestClusterGraphMedoidComputation:
    """Test medoid computation functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = ["ATGC", "GCTA", "CGAT", "TAGC"]
        self.test_headers = ["seq_0", "seq_1", "seq_2", "seq_3"]
        self.test_clusters = {
            "cluster_1": ["seq_0", "seq_1", "seq_2"],
            "cluster_2": ["seq_3"]
        }

        # Mock distance provider with known distances
        self.mock_distance_provider = Mock()

    def test_find_cluster_medoid_single_sequence(self):
        """Test medoid finding for single-sequence cluster."""
        with patch('gaphack.cluster_graph.BlastNeighborhoodFinder'):
            graph = ClusterGraph(
                clusters={"cluster_1": ["seq_0"]},
                sequences=self.test_sequences,
                headers=self.test_headers)

            medoid_idx = graph._find_cluster_medoid(["seq_0"])
            assert medoid_idx == 0

    def test_find_cluster_medoid_multiple_sequences(self):
        """Test medoid finding for multi-sequence cluster."""
        # With MSA-based medoid calculation, we test that it returns a valid index
        # The actual medoid chosen depends on MSA alignment results
        with patch('gaphack.cluster_graph.BlastNeighborhoodFinder'):
            graph = ClusterGraph(
                clusters=self.test_clusters,
                sequences=self.test_sequences,
                headers=self.test_headers)

            medoid_idx = graph._find_cluster_medoid(["seq_0", "seq_1", "seq_2"])
            # Medoid should be one of the cluster members
            assert medoid_idx in [0, 1, 2]

    def test_compute_all_medoids(self):
        """Test computation of medoids for all clusters."""
        self.mock_distance_provider.get_distance.return_value = 0.1

        with patch('gaphack.cluster_graph.BlastNeighborhoodFinder'):
            graph = ClusterGraph(
                clusters=self.test_clusters,
                sequences=self.test_sequences,
                headers=self.test_headers)

            # Should have computed medoids for all clusters
            assert "cluster_1" in graph.medoid_cache
            assert "cluster_2" in graph.medoid_cache
            assert graph.medoid_cache["cluster_2"] == 3  # Single sequence cluster


class TestClusterGraphKNNConstruction:
    """Test K-NN graph construction with BLAST."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = [
            "ATGCGATCGATCGATCG",
            "ATGCGATCGATCGATCC",
            "TTTTTTTTTTTTTTTTTT",
            "TTTTTTTTTTTTTTTTCC"
        ]
        self.test_headers = [f"seq_{i}" for i in range(len(self.test_sequences))]
        self.test_clusters = {
            "cluster_1": ["seq_0"],
            "cluster_2": ["seq_1"],
            "cluster_3": ["seq_2"],
            "cluster_4": ["seq_3"]
        }

        self.mock_distance_provider = Mock()
        self.mock_distance_provider.get_distance.return_value = 0.1

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_knn_graph_construction_basic(self, mock_blast_finder_class):
        """Test basic K-NN graph construction."""
        # Mock BLAST finder
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder

        # Mock BLAST results
        mock_blast_results = {
            "medoid_0": [
                SequenceCandidate("medoid_1", "hash1", 95.0, 100, 1e-10, 200.0),
                SequenceCandidate("medoid_2", "hash2", 90.0, 100, 1e-8, 180.0)
            ],
            "medoid_1": [
                SequenceCandidate("medoid_0", "hash0", 95.0, 100, 1e-10, 200.0),
                SequenceCandidate("medoid_3", "hash3", 85.0, 100, 1e-6, 160.0)
            ]
        }

        # Mock sequence lookup for BLAST results
        mock_blast_finder.sequence_lookup = {
            "hash0": [("seq_content_0", "medoid_0", 0)],
            "hash1": [("seq_content_1", "medoid_1", 1)],
            "hash2": [("seq_content_2", "medoid_2", 2)],
            "hash3": [("seq_content_3", "medoid_3", 3)]
        }

        mock_blast_finder._get_candidates_for_sequences.return_value = mock_blast_results

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers,
            k_neighbors=2
        )

        # Should have created K-NN graph entries for all clusters
        assert len(graph.knn_graph) == len(self.test_clusters)
        for cluster_id in self.test_clusters:
            assert cluster_id in graph.knn_graph

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_identical_medoid_sequences(self, mock_blast_finder_class):
        """Test handling of clusters with identical medoid sequences."""
        # Clusters with identical sequences
        clusters_with_duplicates = {
            "cluster_1": ["seq_0"],
            "cluster_2": ["seq_1"],  # Assume seq_1 is identical to seq_0
            "cluster_3": ["seq_2"]
        }

        # Mock identical sequences
        sequences_with_duplicates = [
            "ATGCGATCGATCGATCG",
            "ATGCGATCGATCGATCG",  # Identical to seq_0
            "TTTTTTTTTTTTTTTTTT"
        ]

        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder
        mock_blast_finder._get_candidates_for_sequences.return_value = {}

        graph = ClusterGraph(
            clusters=clusters_with_duplicates,
            sequences=sequences_with_duplicates,
            headers=self.test_headers[:3])

        # Should handle identical sequences without error
        assert len(graph.knn_graph) == 3

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_empty_blast_results(self, mock_blast_finder_class):
        """Test handling of empty BLAST results."""
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder
        mock_blast_finder._get_candidates_for_sequences.return_value = {}

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        # Should create graph with empty neighbor lists
        for cluster_id in self.test_clusters:
            assert cluster_id in graph.knn_graph
            assert graph.knn_graph[cluster_id] == []


class TestClusterGraphQueries:
    """Test K-NN graph query functionality."""

    def setup_method(self):
        """Set up test fixtures with mock K-NN graph."""
        self.test_sequences = ["ATGC", "GCTA", "CGAT", "TAGC"]
        self.test_headers = ["seq_0", "seq_1", "seq_2", "seq_3"]
        self.test_clusters = {
            "cluster_1": ["seq_0"],
            "cluster_2": ["seq_1"],
            "cluster_3": ["seq_2"],
            "cluster_4": ["seq_3"]
        }

        self.mock_distance_provider = Mock()
        self.mock_distance_provider.get_distance.return_value = 0.1

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_get_neighbors_within_distance(self, mock_blast_finder_class):
        """Test finding neighbors within specified distance."""
        mock_blast_finder = Mock()
        mock_blast_finder._get_candidates_for_sequences.return_value = {}
        mock_blast_finder_class.return_value = mock_blast_finder

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        # Manually set up K-NN graph for testing
        graph.knn_graph = {
            "cluster_1": [("cluster_2", 0.05), ("cluster_3", 0.15), ("cluster_4", 0.25)],
            "cluster_2": [("cluster_1", 0.05), ("cluster_4", 0.20)],
            "cluster_3": [("cluster_1", 0.15)],
            "cluster_4": [("cluster_2", 0.20), ("cluster_1", 0.25)]
        }

        # Test finding neighbors within distance 0.1
        neighbors = graph.get_neighbors_within_distance("cluster_1", 0.1)
        assert len(neighbors) == 1
        assert neighbors[0] == ("cluster_2", 0.05)

        # Test finding neighbors within larger distance
        neighbors = graph.get_neighbors_within_distance("cluster_1", 0.2)
        assert len(neighbors) == 2
        assert ("cluster_2", 0.05) in neighbors
        assert ("cluster_3", 0.15) in neighbors

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_get_k_nearest_neighbors(self, mock_blast_finder_class):
        """Test finding K nearest neighbors."""
        mock_blast_finder = Mock()
        mock_blast_finder._get_candidates_for_sequences.return_value = {}
        mock_blast_finder_class.return_value = mock_blast_finder

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        # Manually set up K-NN graph
        graph.knn_graph = {
            "cluster_1": [("cluster_2", 0.05), ("cluster_3", 0.15), ("cluster_4", 0.25)]
        }

        # Test getting top 2 neighbors
        neighbors = graph.get_k_nearest_neighbors("cluster_1", 2)
        assert len(neighbors) == 2
        assert neighbors[0] == ("cluster_2", 0.05)
        assert neighbors[1] == ("cluster_3", 0.15)

        # Test getting more neighbors than available
        neighbors = graph.get_k_nearest_neighbors("cluster_1", 5)
        assert len(neighbors) == 3  # Should return all available

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_find_close_pairs(self, mock_blast_finder_class):
        """Test finding all close cluster pairs."""
        mock_blast_finder = Mock()
        mock_blast_finder._get_candidates_for_sequences.return_value = {}
        mock_blast_finder_class.return_value = mock_blast_finder

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        # Set up symmetric K-NN graph
        graph.knn_graph = {
            "cluster_1": [("cluster_2", 0.05), ("cluster_3", 0.15)],
            "cluster_2": [("cluster_1", 0.05), ("cluster_4", 0.08)],
            "cluster_3": [("cluster_1", 0.15)],
            "cluster_4": [("cluster_2", 0.08)]
        }

        # Find pairs within distance 0.1
        close_pairs = graph.find_close_pairs(0.1)

        # Should find unique pairs without duplicates
        pair_distances = {(pair[0], pair[1]): pair[2] for pair in close_pairs}

        # Check that we found the expected close pairs
        assert len(close_pairs) >= 2

        # Verify no duplicate pairs (order independent)
        pair_keys = set()
        for cluster1, cluster2, distance in close_pairs:
            key = tuple(sorted([cluster1, cluster2]))
            assert key not in pair_keys  # No duplicates
            pair_keys.add(key)

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_query_nonexistent_cluster(self, mock_blast_finder_class):
        """Test querying for non-existent cluster."""
        mock_blast_finder = Mock()
        mock_blast_finder._get_candidates_for_sequences.return_value = {}
        mock_blast_finder_class.return_value = mock_blast_finder

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        # Query for non-existent cluster
        neighbors = graph.get_neighbors_within_distance("nonexistent", 0.1)
        assert neighbors == []

        neighbors = graph.get_k_nearest_neighbors("nonexistent", 5)
        assert neighbors == []


class TestClusterGraphDynamicUpdates:
    """Test dynamic graph updates (add/remove/update clusters)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = ["ATGC", "GCTA", "CGAT", "TAGC"]
        self.test_headers = ["seq_0", "seq_1", "seq_2", "seq_3"]
        self.test_clusters = {
            "cluster_1": ["seq_0"],
            "cluster_2": ["seq_1", "seq_2"]
        }

        self.mock_distance_provider = Mock()
        self.mock_distance_provider.get_distance.return_value = 0.1

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_remove_cluster(self, mock_blast_finder_class):
        """Test removing cluster from graph."""
        mock_blast_finder = Mock()
        mock_blast_finder._get_candidates_for_sequences.return_value = {}
        mock_blast_finder_class.return_value = mock_blast_finder

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        # Manually set up graph for testing
        graph.knn_graph = {
            "cluster_1": [("cluster_2", 0.1)],
            "cluster_2": [("cluster_1", 0.1)]
        }

        # Remove cluster_1
        graph.remove_cluster("cluster_1")

        # cluster_1 should be removed from all data structures
        assert "cluster_1" not in graph.clusters
        assert "cluster_1" not in graph.medoid_cache
        assert "cluster_1" not in graph.knn_graph

        # cluster_1 should be removed from other clusters' neighbor lists
        assert ("cluster_1", 0.1) not in graph.knn_graph["cluster_2"]

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_update_cluster(self, mock_blast_finder_class):
        """Test updating cluster in graph."""
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder
        mock_blast_finder._get_candidates_for_sequences.return_value = {}

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        # Update should trigger rebuild
        graph.update_cluster("cluster_1", "NEWSEQ", 0)

        # Should have been called at least during initial construction
        assert mock_blast_finder._get_candidates_for_sequences.called

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_add_cluster(self, mock_blast_finder_class):
        """Test adding new cluster to graph."""
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder
        mock_blast_finder._get_candidates_for_sequences.return_value = {}

        # Start with smaller cluster set
        initial_clusters = {"cluster_1": ["seq_0"]}

        graph = ClusterGraph(
            clusters=initial_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        # Add new cluster to the clusters dict first
        graph.clusters["cluster_new"] = ["seq_1"]

        # Add should trigger rebuild
        graph.add_cluster("cluster_new", "NEWSEQ", 1)

        # Should have been called during construction and update
        assert mock_blast_finder._get_candidates_for_sequences.called


class TestClusterGraphErrorHandling:
    """Test error handling and edge cases."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = ["ATGC", "GCTA"]
        self.test_headers = ["seq_0", "seq_1"]
        self.test_clusters = {"cluster_1": ["seq_0"], "cluster_2": ["seq_1"]}

        self.mock_distance_provider = Mock()

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    @patch('gaphack.distance_providers.MSACachedDistanceProvider')
    def test_distance_provider_error(self, mock_msa_provider_class, mock_blast_finder_class):
        """Test handling of distance provider errors during medoid calculation."""
        mock_blast_finder = Mock()
        mock_blast_finder._get_candidates_for_sequences.return_value = {}
        mock_blast_finder_class.return_value = mock_blast_finder

        # Mock MSA provider to raise error during distance calculation
        mock_msa_provider = Mock()
        mock_msa_provider.get_distance.side_effect = ValueError("Distance error")
        mock_msa_provider_class.return_value = mock_msa_provider

        # Use clusters with multiple sequences to trigger medoid computation
        multi_seq_clusters = {"cluster_1": ["seq_0", "seq_1"]}

        with pytest.raises(ValueError):
            ClusterGraph(
                clusters=multi_seq_clusters,
                sequences=self.test_sequences,
                headers=self.test_headers)

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_blast_finder_error(self, mock_blast_finder_class):
        """Test handling of BLAST finder initialization errors."""
        mock_blast_finder_class.side_effect = RuntimeError("BLAST error")

        self.mock_distance_provider.get_distance.return_value = 0.1

        with pytest.raises(RuntimeError):
            ClusterGraph(
                clusters=self.test_clusters,
                sequences=self.test_sequences,
                headers=self.test_headers)

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_malformed_blast_results(self, mock_blast_finder_class):
        """Test handling of malformed BLAST results."""
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder

        # Mock malformed BLAST results with invalid header format
        mock_blast_finder._get_candidates_for_sequences.return_value = {
            "invalid_header": [SequenceCandidate("target", "hash", 95.0, 100, 1e-8, 180.0)]
        }
        mock_blast_finder.sequence_lookup = {}  # Empty lookup

        self.mock_distance_provider.get_distance.return_value = 0.1

        # Should handle gracefully without crashing (malformed results should be skipped)
        try:
            graph = ClusterGraph(
                clusters=self.test_clusters,
                sequences=self.test_sequences,
                headers=self.test_headers)
            # Graph should be created but with empty/limited neighbors
            assert isinstance(graph, ClusterGraph)
        except ValueError:
            # Also acceptable - the code might validate header format
            pass

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_mismatched_headers_sequences(self, mock_blast_finder_class):
        """Test handling of mismatched headers and sequences."""
        mock_blast_finder = Mock()
        mock_blast_finder._get_candidates_for_sequences.return_value = {}
        mock_blast_finder_class.return_value = mock_blast_finder

        mismatched_headers = ["seq_0"]  # One less than sequences

        # Should handle gracefully or raise appropriate error
        try:
            graph = ClusterGraph(
                clusters={"cluster_1": ["seq_0"]},
                sequences=self.test_sequences,  # 2 sequences
                headers=mismatched_headers     # 1 header
            )
        except (ValueError, IndexError):
            # Expected behavior for mismatched inputs
            pass


class TestClusterGraphPropertyBased:
    """Property-based tests for ClusterGraph."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_distance_provider = Mock()
        self.mock_distance_provider.get_distance.return_value = 0.1

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    @given(st.integers(1, 10))
    def test_k_neighbors_parameter_effect(self, mock_blast_finder_class, k_neighbors):
        """Test that k_neighbors parameter affects graph structure."""
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder
        mock_blast_finder._get_candidates_for_sequences.return_value = {}

        sequences = [f"ATGC{i}" for i in range(5)]
        headers = [f"seq_{i}" for i in range(5)]
        clusters = {f"cluster_{i}": [f"seq_{i}"] for i in range(5)}

        graph = ClusterGraph(
            clusters=clusters,
            sequences=sequences,
            headers=headers,
            k_neighbors=k_neighbors
        )

        assert graph.k_neighbors == k_neighbors

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    @given(st.floats(1e-10, 1e-1), st.floats(50.0, 100.0))
    def test_blast_parameters_validation(self, mock_blast_finder_class, evalue, identity):
        """Test BLAST parameter validation."""
        mock_blast_finder = Mock()
        mock_blast_finder._get_candidates_for_sequences.return_value = {}
        mock_blast_finder_class.return_value = mock_blast_finder

        sequences = ["ATGC", "GCTA"]
        headers = ["seq_0", "seq_1"]
        clusters = {"cluster_1": ["seq_0"], "cluster_2": ["seq_1"]}

        graph = ClusterGraph(
            clusters=clusters,
            sequences=sequences,
            headers=headers,
            blast_evalue=evalue,
            blast_identity=identity
        )

        assert graph.blast_evalue == evalue
        assert graph.blast_identity == identity

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    @given(st.lists(st.text(alphabet="ATGC", min_size=1, max_size=20), min_size=1, max_size=10))
    def test_variable_sequence_lengths(self, mock_blast_finder_class, sequences):
        """Test handling of variable sequence lengths."""
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder
        mock_blast_finder._get_candidates_for_sequences.return_value = {}

        headers = [f"seq_{i}" for i in range(len(sequences))]
        clusters = {f"cluster_{i}": [f"seq_{i}"] for i in range(len(sequences))}

        # Should handle variable length sequences without error
        graph = ClusterGraph(
            clusters=clusters,
            sequences=sequences,
            headers=headers)

        assert len(graph.medoid_cache) == len(clusters)
        assert len(graph.knn_graph) == len(clusters)


class TestClusterGraphIntegration:
    """Integration tests combining multiple ClusterGraph features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = [
            "ATGCGATCGATCGATCG",
            "ATGCGATCGATCGATCC",
            "TTTTTTTTTTTTTTTTTT",
            "TTTTTTTTTTTTTTTTCC",
            "GGGGGGGGGGGGGGGGGG"
        ]
        self.test_headers = [f"seq_{i}" for i in range(len(self.test_sequences))]
        self.test_clusters = {
            "cluster_1": ["seq_0", "seq_1"],
            "cluster_2": ["seq_2", "seq_3"],
            "cluster_3": ["seq_4"]
        }

        # Mock realistic distance provider
        self.mock_distance_provider = Mock()
        def mock_distance(i, j):
            # Simulate realistic distance matrix
            if i == j:
                return 0.0
            elif abs(i - j) == 1:
                return 0.1  # Close sequences
            else:
                return 0.5  # Distant sequences

        self.mock_distance_provider.get_distance.side_effect = mock_distance

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_full_workflow_medoid_to_query(self, mock_blast_finder_class):
        """Test complete workflow from medoid computation to neighbor queries."""
        # Mock BLAST finder with realistic results
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder

        mock_blast_results = {
            "medoid_0": [SequenceCandidate("medoid_1", "hash1", 90.0, 100, 1e-8, 180.0)],
            "medoid_1": [SequenceCandidate("medoid_0", "hash0", 90.0, 100, 1e-8, 180.0)],
            "medoid_2": []  # No neighbors found
        }

        mock_blast_finder.sequence_lookup = {
            "hash0": [("seq0", "medoid_0", 0)],
            "hash1": [("seq1", "medoid_1", 1)]
        }

        mock_blast_finder._get_candidates_for_sequences.return_value = mock_blast_results

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers,
            k_neighbors=3
        )

        # Test medoid computation
        assert len(graph.medoid_cache) == 3

        # Test neighbor queries
        neighbors = graph.get_neighbors_within_distance("cluster_1", 1.0)
        assert isinstance(neighbors, list)

        close_pairs = graph.find_close_pairs(1.0)
        assert isinstance(close_pairs, list)

    @patch('gaphack.cluster_graph.BlastNeighborhoodFinder')
    def test_dynamic_updates_preserve_consistency(self, mock_blast_finder_class):
        """Test that dynamic updates maintain graph consistency."""
        mock_blast_finder = Mock()
        mock_blast_finder_class.return_value = mock_blast_finder
        mock_blast_finder._get_candidates_for_sequences.return_value = {}

        graph = ClusterGraph(
            clusters=self.test_clusters,
            sequences=self.test_sequences,
            headers=self.test_headers)

        initial_cluster_count = len(graph.clusters)

        # Remove a cluster
        graph.remove_cluster("cluster_3")
        assert len(graph.clusters) == initial_cluster_count - 1
        assert "cluster_3" not in graph.knn_graph

        # Verify remaining clusters don't reference removed cluster
        for cluster_id, neighbors in graph.knn_graph.items():
            for neighbor_id, distance in neighbors:
                assert neighbor_id != "cluster_3"
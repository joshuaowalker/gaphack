"""
Tests for the core clustering algorithm.
"""

import pytest
import numpy as np
from gaphack.core import GapOptimizedClustering


class TestGapOptimizedClustering:
    """Test suite for GapOptimizedClustering class."""
    
    def test_initialization(self):
        """Test that the clustering object initializes correctly."""
        clustering = GapOptimizedClustering()
        assert clustering.min_split == 0.005
        assert clustering.max_lump == 0.02
        assert clustering.target_percentile == 95
    
    def test_custom_parameters(self):
        """Test initialization with custom parameters."""
        clustering = GapOptimizedClustering(
            min_split=0.01,
            max_lump=0.03,
            target_percentile=90,
            num_threads=0  # Single-process mode for testing
        )
        assert clustering.min_split == 0.01
        assert clustering.max_lump == 0.03
        assert clustering.target_percentile == 90
        assert clustering.num_threads == 0
    
    def test_simple_clustering(self):
        """Test clustering with a simple distance matrix."""
        # Create a simple distance matrix with clear clusters
        # Cluster 1: sequences 0, 1 (close to each other)
        # Cluster 2: sequences 2, 3 (close to each other)
        # Large gap between clusters
        distance_matrix = np.array([
            [0.00, 0.01, 0.50, 0.51],  # seq 0
            [0.01, 0.00, 0.52, 0.53],  # seq 1
            [0.50, 0.52, 0.00, 0.01],  # seq 2
            [0.51, 0.53, 0.01, 0.00],  # seq 3
        ])
        
        clustering = GapOptimizedClustering()
        clusters, singletons, metrics = clustering.cluster(distance_matrix)
        
        # Should find 2 clusters
        assert len(clusters) == 2
        assert len(singletons) == 0
        
        # Check that sequences are correctly grouped
        cluster_sets = [set(c) for c in clusters]
        assert {0, 1} in cluster_sets
        assert {2, 3} in cluster_sets
    
    def test_singletons(self):
        """Test that isolated sequences become singletons."""
        # Create a distance matrix with one clear cluster and one isolated sequence
        distance_matrix = np.array([
            [0.00, 0.01, 0.02, 0.90],  # seq 0 (cluster)
            [0.01, 0.00, 0.01, 0.91],  # seq 1 (cluster)
            [0.02, 0.01, 0.00, 0.92],  # seq 2 (cluster)
            [0.90, 0.91, 0.92, 0.00],  # seq 3 (singleton)
        ])
        
        clustering = GapOptimizedClustering()
        clusters, singletons, metrics = clustering.cluster(distance_matrix)
        
        # Should find 1 cluster and 1 singleton
        assert len(clusters) == 1
        assert len(singletons) == 1
        assert set(clusters[0]) == {0, 1, 2}
        assert singletons[0] == 3
    
    def test_percentile_linkage(self):
        """Test percentile complete linkage distance calculation."""
        clustering = GapOptimizedClustering()
        
        # Simple distance matrix
        distance_matrix = np.array([
            [0.0, 0.1, 0.5, 0.6],
            [0.1, 0.0, 0.4, 0.7],
            [0.5, 0.4, 0.0, 0.2],
            [0.6, 0.7, 0.2, 0.0],
        ])
        
        # Calculate distance between clusters {0, 1} and {2, 3}
        cluster1 = {0, 1}
        cluster2 = {2, 3}
        
        # At 95th percentile, we expect close to the maximum distance
        dist = clustering._percentile_complete_linkage_distance(
            cluster1, cluster2, distance_matrix, 95.0
        )
        
        # The distances are: 0.5, 0.6, 0.4, 0.7
        # Sorted: 0.4, 0.5, 0.6, 0.7
        # 95th percentile should be close to 0.7
        assert 0.6 <= dist <= 0.7
    
    def test_gap_calculation(self):
        """Test that clustering produces meaningful results with a clear gap."""
        clustering = GapOptimizedClustering(num_threads=0)  # Single-process for testing
        
        # Create a distance matrix with a clear gap structure
        distance_matrix = np.array([
            [0.00, 0.01, 0.10, 0.11],  # Clear gap between
            [0.01, 0.00, 0.12, 0.13],  # intra (0.01) and
            [0.10, 0.12, 0.00, 0.01],  # inter (0.10+)
            [0.11, 0.13, 0.01, 0.00],
        ])
        
        clusters, singletons, metrics = clustering.cluster(distance_matrix)
        
        # Should produce 2 clusters with the clear gap structure
        assert len(clusters) == 2
        assert len(singletons) == 0
        assert metrics['best_config']['gap_size'] > 0
    
    def test_empty_distance_matrix(self):
        """Test handling of empty distance matrix."""
        clustering = GapOptimizedClustering(num_threads=0)  # Single-process for testing
        distance_matrix = np.array([]).reshape(0, 0)
        
        clusters, singletons, metrics = clustering.cluster(distance_matrix)
        
        assert len(clusters) == 0
        assert len(singletons) == 0
    
    def test_single_sequence(self):
        """Test clustering with a single sequence."""
        clustering = GapOptimizedClustering(num_threads=0)  # Single-process for testing
        distance_matrix = np.array([[0.0]])
        
        clusters, singletons, metrics = clustering.cluster(distance_matrix)
        
        assert len(clusters) == 0
        assert len(singletons) == 1
        assert singletons[0] == 0
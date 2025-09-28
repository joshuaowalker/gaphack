"""
Tests for the core clustering algorithm.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
from gaphack.core import GapOptimizedClustering, DistanceCache, GapCalculator, PersistentWorker


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


class TestGapCalculator:
    """Comprehensive whitebox tests for GapCalculator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.gap_calculator = GapCalculator(target_percentile=95)

    def test_initialization(self):
        """Test GapCalculator initialization with different percentiles."""
        calculator_95 = GapCalculator(95)
        assert calculator_95.target_percentile == 95

        calculator_90 = GapCalculator(90)
        assert calculator_90.target_percentile == 90

        # Test default
        default_calculator = GapCalculator()
        assert default_calculator.target_percentile == 95

    def test_empty_clusters_percentile_distance(self):
        """Test percentile distance calculation with empty clusters."""
        # Create minimal distance matrix
        distance_matrix = np.array([[0.0, 0.1], [0.1, 0.0]])
        cache = DistanceCache(distance_matrix)

        # Empty clusters should return 0.0
        empty_cluster1 = set()
        empty_cluster2 = set()

        distance = self.gap_calculator.calculate_percentile_cluster_distance(
            empty_cluster1, empty_cluster2, cache
        )
        assert distance == 0.0

    def test_single_element_clusters_percentile_distance(self):
        """Test percentile distance calculation with single-element clusters."""
        distance_matrix = np.array([
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.3],
            [0.2, 0.3, 0.0]
        ])
        cache = DistanceCache(distance_matrix)

        cluster1 = {0}
        cluster2 = {1}

        # Single element clusters only have inter-cluster distance
        distance = self.gap_calculator.calculate_percentile_cluster_distance(
            cluster1, cluster2, cache
        )
        # Should return the distance between elements 0 and 1
        assert distance == 0.1

    def test_percentile_calculation_precision(self):
        """Test that percentile calculations are precise for known cases."""
        # Create distance matrix with known values
        distance_matrix = np.array([
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0]
        ])
        cache = DistanceCache(distance_matrix)

        # Test with clusters that will produce specific distance sets
        cluster1 = {0, 1}  # intra-distances: [0.1]
        cluster2 = {2, 3}  # intra-distances: [0.6]
        # inter-distances: [0.2, 0.3, 0.4, 0.5]
        # Combined sorted: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

        distance = self.gap_calculator.calculate_percentile_cluster_distance(
            cluster1, cluster2, cache
        )

        # 95th percentile of 6 values (index 5.7 → index 5) should be 0.6
        expected_95th_percentile = 0.6
        assert distance == expected_95th_percentile

    def test_percentile_boundary_conditions(self):
        """Test percentile calculation at boundary conditions."""
        distance_matrix = np.array([
            [0.0, 0.1, 0.2],
            [0.1, 0.0, 0.3],
            [0.2, 0.3, 0.0]
        ])
        cache = DistanceCache(distance_matrix)

        # Test different percentiles
        for percentile in [0, 25, 50, 75, 95, 100]:
            calculator = GapCalculator(target_percentile=percentile)

            cluster1 = {0}
            cluster2 = {1, 2}

            distance = calculator.calculate_percentile_cluster_distance(
                cluster1, cluster2, cache
            )

            # Distance should be non-negative and finite
            assert distance >= 0
            assert np.isfinite(distance)

    @given(
        percentile=st.integers(min_value=1, max_value=100),
        matrix_size=st.integers(min_value=2, max_value=8)
    )
    def test_percentile_calculation_properties(self, percentile, matrix_size):
        """Property-based test for percentile calculation mathematical properties."""
        # Generate random symmetric distance matrix
        np.random.seed(42)  # For reproducibility
        random_matrix = np.random.rand(matrix_size, matrix_size)
        # Make symmetric and zero diagonal
        distance_matrix = (random_matrix + random_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        cache = DistanceCache(distance_matrix)
        calculator = GapCalculator(target_percentile=percentile)

        # Generate random non-empty clusters
        cluster1_size = max(1, matrix_size // 3)
        cluster2_size = max(1, matrix_size // 3)

        all_indices = list(range(matrix_size))
        np.random.shuffle(all_indices)

        cluster1 = set(all_indices[:cluster1_size])
        cluster2 = set(all_indices[cluster1_size:cluster1_size + cluster2_size])

        # Skip if clusters overlap or are empty
        if cluster1.intersection(cluster2) or not cluster1 or not cluster2:
            return

        distance = calculator.calculate_percentile_cluster_distance(
            cluster1, cluster2, cache
        )

        # Mathematical properties that should always hold
        assert distance >= 0, "Distance must be non-negative"
        assert np.isfinite(distance), "Distance must be finite"

        # Distance should be bounded by the maximum distance in the matrix
        max_distance = np.max(distance_matrix)
        assert distance <= max_distance, f"Distance {distance} exceeds max {max_distance}"


class TestDistanceCache:
    """Comprehensive whitebox tests for DistanceCache class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.distance_matrix = np.array([
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0]
        ])
        self.cache = DistanceCache(self.distance_matrix)

    def test_cache_initialization(self):
        """Test DistanceCache initialization."""
        assert np.array_equal(self.cache.distance_matrix, self.distance_matrix)
        assert self.cache.intra_cache == {}
        assert self.cache.inter_cache == {}
        assert self.cache.global_intra_distances is None

    def test_intra_distances_single_element(self):
        """Test intra-cluster distances for single element cluster."""
        cluster = {0}
        distances = self.cache.get_intra_distances(cluster)

        # Single element cluster has no intra-distances
        assert distances == ()

    def test_intra_distances_two_elements(self):
        """Test intra-cluster distances for two element cluster."""
        cluster = {0, 1}
        distances = self.cache.get_intra_distances(cluster)

        # Two element cluster has one distance
        assert distances == (0.1,)

    def test_intra_distances_multiple_elements(self):
        """Test intra-cluster distances for larger cluster."""
        cluster = {0, 1, 2}
        distances = self.cache.get_intra_distances(cluster)

        # Three element cluster has three pairwise distances
        # Distances: (0,1)=0.1, (0,2)=0.2, (1,2)=0.4
        # Sorted: [0.1, 0.2, 0.4]
        assert distances == (0.1, 0.2, 0.4)

    def test_intra_distances_caching(self):
        """Test that intra-distances are properly cached."""
        cluster = {0, 1, 2}

        # First call should compute and cache
        distances1 = self.cache.get_intra_distances(cluster)
        assert frozenset(cluster) in self.cache.intra_cache

        # Second call should return cached result (same object)
        distances2 = self.cache.get_intra_distances(cluster)
        assert distances1 is distances2

    def test_inter_distances_basic(self):
        """Test basic inter-cluster distance calculation."""
        cluster1 = {0}
        cluster2 = {1}

        distances = self.cache.get_inter_distances(cluster1, cluster2)

        # Single element clusters have one inter-distance
        assert distances == (0.1,)

    def test_inter_distances_multiple_elements(self):
        """Test inter-cluster distances with multiple elements."""
        cluster1 = {0, 1}
        cluster2 = {2, 3}

        distances = self.cache.get_inter_distances(cluster1, cluster2)

        # Expected distances: (0,2)=0.2, (0,3)=0.3, (1,2)=0.4, (1,3)=0.5
        # Sorted: [0.2, 0.3, 0.4, 0.5]
        assert distances == (0.2, 0.3, 0.4, 0.5)

    def test_inter_distances_caching(self):
        """Test that inter-distances are properly cached."""
        cluster1 = {0, 1}
        cluster2 = {2, 3}

        # First call should compute and cache
        distances1 = self.cache.get_inter_distances(cluster1, cluster2)
        cache_key = frozenset([frozenset(cluster1), frozenset(cluster2)])
        assert cache_key in self.cache.inter_cache

        # Second call should return cached result
        distances2 = self.cache.get_inter_distances(cluster1, cluster2)
        assert distances1 is distances2

        # Reverse order should also hit cache
        distances3 = self.cache.get_inter_distances(cluster2, cluster1)
        assert distances1 is distances3

    def test_cache_coherency_with_nan_values(self):
        """Test cache behavior with NaN values in distance matrix."""
        # Create matrix with NaN values
        matrix_with_nan = np.array([
            [0.0, 0.1, np.nan],
            [0.1, 0.0, 0.2],
            [np.nan, 0.2, 0.0]
        ])
        cache = DistanceCache(matrix_with_nan)

        cluster = {0, 2}
        distances = cache.get_intra_distances(cluster)

        # NaN values should be filtered out
        # Only valid distances should remain (none in this case)
        assert distances == ()

    def test_cache_memory_efficiency(self):
        """Test that cache uses tuples for memory efficiency."""
        cluster = {0, 1, 2}
        distances = self.cache.get_intra_distances(cluster)

        # Returned distances should be immutable tuple
        assert isinstance(distances, tuple)

        # Cached value should also be tuple
        cached_value = self.cache.intra_cache[frozenset(cluster)]
        assert isinstance(cached_value, tuple)

    @given(
        cluster_size=st.integers(min_value=1, max_value=6),
        matrix_size=st.integers(min_value=3, max_value=10)
    )
    def test_distance_cache_properties(self, cluster_size, matrix_size):
        """Property-based test for distance cache mathematical properties."""
        assume(cluster_size <= matrix_size)

        # Generate random symmetric distance matrix
        np.random.seed(42)
        random_matrix = np.random.rand(matrix_size, matrix_size)
        distance_matrix = (random_matrix + random_matrix.T) / 2
        np.fill_diagonal(distance_matrix, 0)

        cache = DistanceCache(distance_matrix)

        # Generate random cluster
        indices = list(range(matrix_size))
        np.random.shuffle(indices)
        cluster = set(indices[:cluster_size])

        distances = cache.get_intra_distances(cluster)

        # Properties that should always hold
        assert isinstance(distances, tuple), "Distances should be tuple"
        assert all(d >= 0 for d in distances), "All distances should be non-negative"
        assert distances == tuple(sorted(distances)), "Distances should be sorted"

        # Number of distances should follow combinatorial formula
        expected_count = max(0, cluster_size * (cluster_size - 1) // 2)
        assert len(distances) == expected_count, f"Expected {expected_count} distances, got {len(distances)}"


class TestIncrementalGapCalculation:
    """Tests for incremental gap calculation accuracy and performance."""

    def setup_method(self):
        """Set up test fixtures."""
        self.distance_matrix = np.array([
            [0.0, 0.01, 0.50, 0.51],  # Clear two-cluster structure
            [0.01, 0.0, 0.52, 0.53],
            [0.50, 0.52, 0.0, 0.01],
            [0.51, 0.53, 0.01, 0.0]
        ])
        self.cache = DistanceCache(self.distance_matrix)
        self.gap_calculator = GapCalculator(target_percentile=95)

    def test_incremental_vs_full_calculation_consistency(self):
        """Test that incremental calculation matches full recalculation."""
        cluster1 = {0, 1}
        cluster2 = {2, 3}

        # Calculate using incremental method
        incremental_distance = self.gap_calculator.calculate_percentile_cluster_distance(
            cluster1, cluster2, self.cache
        )

        # Calculate using direct index-based approach (matching incremental method)
        intra1 = self.cache.get_intra_distances(cluster1)
        intra2 = self.cache.get_intra_distances(cluster2)
        inter = self.cache.get_inter_distances(cluster1, cluster2)

        all_distances = sorted(list(intra1) + list(intra2) + list(inter))

        # Use the same index calculation as the incremental method
        total_distances = len(all_distances)
        percentile_idx = int(total_distances * (95 / 100.0))
        if percentile_idx >= total_distances:
            percentile_idx = total_distances - 1

        full_distance = all_distances[percentile_idx]

        # Should be identical (within floating point precision)
        assert abs(incremental_distance - full_distance) < 1e-10

    def test_incremental_calculation_edge_cases(self):
        """Test incremental calculation with edge cases."""
        # Test with empty distance lists
        cluster1 = {0}
        cluster2 = {1}

        distance = self.gap_calculator.calculate_percentile_cluster_distance(
            cluster1, cluster2, self.cache
        )

        # Single inter-distance should be returned as-is
        assert distance == 0.01

    def test_percentile_index_calculation_accuracy(self):
        """Test that percentile index calculation handles edge cases correctly."""
        # Test with known distance counts
        cluster1 = {0, 1, 2}  # 3 intra-distances: (0,1), (0,2), (1,2)
        cluster2 = {3}        # 0 intra-distances
        # 3 inter-distances: (0,3), (1,3), (2,3)
        # Total: 6 distances

        distance = self.gap_calculator.calculate_percentile_cluster_distance(
            cluster1, cluster2, self.cache
        )

        # Get all distances and manually calculate 95th percentile
        intra1 = list(self.cache.get_intra_distances(cluster1))
        inter = list(self.cache.get_inter_distances(cluster1, cluster2))
        all_distances = sorted(intra1 + inter)

        # 95th percentile of 6 items: index = int(6 * 0.95) = 5, so all_distances[5]
        expected = all_distances[5] if len(all_distances) > 5 else all_distances[-1]

        assert distance == expected


class TestPersistentWorker:
    """Comprehensive tests for PersistentWorker multiprocessing class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.distance_matrix = np.array([
            [0.0, 0.1, 0.2, 0.3],
            [0.1, 0.0, 0.4, 0.5],
            [0.2, 0.4, 0.0, 0.6],
            [0.3, 0.5, 0.6, 0.0]
        ])
        self.worker = PersistentWorker(
            distance_matrix=self.distance_matrix,
            min_split=0.01,
            max_lump=0.05,
            target_percentile=95
        )

    def test_worker_initialization(self):
        """Test PersistentWorker initialization."""
        assert np.array_equal(self.worker.distance_matrix, self.distance_matrix)
        assert self.worker.min_split == 0.01
        assert self.worker.max_lump == 0.05
        assert self.worker.target_percentile == 95
        assert isinstance(self.worker.cache, DistanceCache)
        assert isinstance(self.worker.gap_calculator, GapCalculator)
        assert self.worker.current_clusters is None

    def test_offset_to_pair_conversion(self):
        """Test mathematical offset to (i,j) pair conversion."""
        n_clusters = 4

        # Test specific known conversions
        test_cases = [
            (0, (0, 1)),  # First pair
            (1, (0, 2)),  # Second pair
            (2, (0, 3)),  # Third pair
            (3, (1, 2)),  # Fourth pair
            (4, (1, 3)),  # Fifth pair
            (5, (2, 3)),  # Sixth pair (last for n=4)
        ]

        for offset, expected_pair in test_cases:
            i, j = self.worker.offset_to_pair(offset, n_clusters)
            assert (i, j) == expected_pair, f"Offset {offset} should map to {expected_pair}, got ({i}, {j})"

    def test_offset_to_pair_mathematical_properties(self):
        """Test mathematical properties of offset to pair conversion."""
        for n_clusters in range(2, 8):
            max_pairs = n_clusters * (n_clusters - 1) // 2

            for offset in range(max_pairs):
                i, j = self.worker.offset_to_pair(offset, n_clusters)

                # Mathematical properties that should always hold
                assert 0 <= i < n_clusters, f"i={i} out of range for n={n_clusters}"
                assert 0 <= j < n_clusters, f"j={j} out of range for n={n_clusters}"
                assert i < j, f"i={i} should be less than j={j} for upper triangle"

    def test_offset_to_pair_invalid_inputs(self):
        """Test offset to pair conversion with invalid inputs."""
        n_clusters = 4
        max_pairs = n_clusters * (n_clusters - 1) // 2  # 6 pairs for n=4

        # Test negative offset
        with pytest.raises(ValueError, match="Offset must be non-negative"):
            self.worker.offset_to_pair(-1, n_clusters)

        # Test offset beyond maximum
        with pytest.raises(ValueError, match="Invalid offset"):
            self.worker.offset_to_pair(max_pairs, n_clusters)

    def test_offset_to_pair_coverage(self):
        """Test that offset conversion covers all upper triangle pairs exactly once."""
        n_clusters = 5
        max_pairs = n_clusters * (n_clusters - 1) // 2

        # Generate all pairs from offsets
        generated_pairs = set()
        for offset in range(max_pairs):
            i, j = self.worker.offset_to_pair(offset, n_clusters)
            generated_pairs.add((i, j))

        # Generate expected upper triangle pairs
        expected_pairs = set()
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                expected_pairs.add((i, j))

        # Should be identical sets
        assert generated_pairs == expected_pairs, "Offset conversion should cover all upper triangle pairs exactly once"

    def test_update_clusters_functionality(self):
        """Test cluster state update functionality."""
        clusters_list = [[0, 1], [2, 3]]

        self.worker.update_clusters(clusters_list)

        # Check that clusters were converted to sets
        assert self.worker.current_clusters == [{0, 1}, {2, 3}]

        # Verify that global caches were refreshed (we can't directly test cache state
        # but we can verify the method was called without error)
        assert self.worker.current_clusters is not None

    @given(
        n_clusters=st.integers(min_value=2, max_value=10),
        offset_fraction=st.floats(min_value=0.0, max_value=0.99)
    )
    def test_offset_conversion_properties(self, n_clusters, offset_fraction):
        """Property-based test for offset conversion mathematical properties."""
        max_pairs = n_clusters * (n_clusters - 1) // 2
        offset = int(offset_fraction * max_pairs)

        i, j = self.worker.offset_to_pair(offset, n_clusters)

        # Properties that should always hold
        assert isinstance(i, int), "i should be integer"
        assert isinstance(j, int), "j should be integer"
        assert 0 <= i < n_clusters, f"i={i} out of range"
        assert 0 <= j < n_clusters, f"j={j} out of range"
        assert i < j, f"Should maintain upper triangle property: i={i} < j={j}"


class TestMultiprocessingIntegration:
    """Integration tests for multiprocessing components."""

    def setup_method(self):
        """Set up test fixtures."""
        self.distance_matrix = np.array([
            [0.0, 0.01, 0.50, 0.51],  # Two clear clusters
            [0.01, 0.0, 0.52, 0.53],
            [0.50, 0.52, 0.0, 0.01],
            [0.51, 0.53, 0.01, 0.0]
        ])

    def test_single_vs_multiprocess_consistency(self):
        """Test that single-process and multiprocess modes produce identical results."""
        # Single-process mode
        clustering_single = GapOptimizedClustering(num_threads=0)
        clusters_single, singletons_single, metrics_single = clustering_single.cluster(self.distance_matrix)

        # Multiprocess mode (if available)
        clustering_multi = GapOptimizedClustering(num_threads=2)
        clusters_multi, singletons_multi, metrics_multi = clustering_multi.cluster(self.distance_matrix)

        # Results should be identical
        assert len(clusters_single) == len(clusters_multi)
        assert len(singletons_single) == len(singletons_multi)

        # Convert to sets for comparison (order may differ)
        clusters_single_sets = [set(c) for c in clusters_single]
        clusters_multi_sets = [set(c) for c in clusters_multi]

        # Should contain the same clusters
        for cluster in clusters_single_sets:
            assert cluster in clusters_multi_sets

        for cluster in clusters_multi_sets:
            assert cluster in clusters_single_sets

    def test_workload_distribution_coverage(self):
        """Test that workload distribution covers all cluster pairs."""
        n_clusters = 6
        max_pairs = n_clusters * (n_clusters - 1) // 2
        num_workers = 3

        worker = PersistentWorker(
            distance_matrix=np.eye(n_clusters),  # Dummy matrix
            min_split=0.01,
            max_lump=0.05,
            target_percentile=95
        )

        # Track which pairs each worker would process
        all_processed_pairs = set()

        for worker_id in range(num_workers):
            worker_pairs = set()

            # Simulate workload distribution
            for global_offset in range(worker_id, max_pairs, num_workers):
                i, j = worker.offset_to_pair(global_offset, n_clusters)
                worker_pairs.add((i, j))

            # No overlap between workers
            assert not worker_pairs.intersection(all_processed_pairs)
            all_processed_pairs.update(worker_pairs)

        # All pairs should be covered
        expected_pairs = set()
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                expected_pairs.add((i, j))

        assert all_processed_pairs == expected_pairs

    def test_worker_persistence_benefits(self):
        """Test that worker persistence provides caching benefits."""
        worker = PersistentWorker(
            distance_matrix=self.distance_matrix,
            min_split=0.01,
            max_lump=0.05,
            target_percentile=95
        )

        # Update clusters to initialize caches
        clusters_list = [[0, 1], [2, 3]]
        worker.update_clusters(clusters_list)

        # Access cache multiple times - should hit cache after first access
        cluster1 = {0, 1}

        # First access - computes and caches
        distances1 = worker.cache.get_intra_distances(cluster1)

        # Second access - should return cached result (same object)
        distances2 = worker.cache.get_intra_distances(cluster1)

        # Should be the same cached object
        assert distances1 is distances2

    def test_error_handling_in_multiprocessing_context(self):
        """Test error handling in multiprocessing context."""
        # Test with invalid distance matrix
        invalid_matrix = np.array([[0.0, -1.0], [-1.0, 0.0]])  # Negative distances

        clustering = GapOptimizedClustering(num_threads=0)  # Single process for clear error

        # Should handle gracefully (implementation may vary)
        try:
            clusters, singletons, metrics = clustering.cluster(invalid_matrix)
            # If it succeeds, verify results are reasonable
            assert isinstance(clusters, list)
            assert isinstance(singletons, list)
            assert isinstance(metrics, dict)
        except (ValueError, np.linalg.LinAlgError):
            # If it fails, that's also acceptable for invalid input
            pass
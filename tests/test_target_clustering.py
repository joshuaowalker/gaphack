"""
Tests for target mode clustering functionality.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

from gaphack.target_clustering import TargetModeClustering


class TestTargetModeClustering:

    def setup_method(self):
        """Set up test fixtures."""
        self.clustering = TargetModeClustering(
            min_split=0.01,
            max_lump=0.05,
            target_percentile=95,
            show_progress=False
        )

    def _generate_test_sequences(self, n: int):
        """Generate test sequences for a given matrix size."""
        base_sequences = ["ATCG", "GCTA", "CGAT", "TAGC", "AGTC", "CTAG", "GATC", "TCGA"]
        return [base_sequences[i % len(base_sequences)] + str(i) for i in range(n)]
    
    def test_init(self):
        """Test TargetModeClustering initialization."""
        assert self.clustering.min_split == 0.01
        assert self.clustering.max_lump == 0.05
        assert self.clustering.target_percentile == 95
        assert self.clustering.show_progress is False
    
    def test_single_target_simple_case(self):
        """Test target clustering with single target and simple distance matrix."""
        # Create a simple 4x4 distance matrix
        # Target sequence (index 0) is closest to sequence 1 (distance 0.02)
        # then to sequence 2 (distance 0.04), then sequence 3 (distance 0.06)
        distance_matrix = np.array([
            [0.0,  0.02, 0.04, 0.06],  # target sequence
            [0.02, 0.0,  0.07, 0.08],  # closest to target
            [0.04, 0.07, 0.0,  0.09],  # second closest to target  
            [0.06, 0.08, 0.09, 0.0]    # farthest from target
        ])
        
        target_indices = [0]
        
        sequences = self._generate_test_sequences(len(distance_matrix))
        target_cluster, remaining, metrics = self.clustering.cluster(distance_matrix, target_indices, sequences)
        
        # Should include target (0), closest sequence (1), and second closest (2)
        # but not farthest (3) since it exceeds max_lump threshold
        assert 0 in target_cluster  # target sequence always included
        assert 1 in target_cluster  # closest sequence should be included  
        assert len(target_cluster) >= 2  # at least target + one other
        
        # Check that remaining sequences are those not in target cluster
        expected_remaining = set(range(4)) - set(target_cluster)
        assert set(remaining) == expected_remaining
        
        # Check metrics structure
        assert 'best_config' in metrics
        assert 'gap_history' in metrics
        assert metrics['best_config']['gap_size'] >= 0
    
    def test_multiple_target_seeds(self):
        """Test target clustering with multiple seed sequences."""
        # 5x5 matrix where sequences 0 and 1 are target seeds
        # and sequence 2 is close to both (should be merged)
        distance_matrix = np.array([
            [0.0,  0.01, 0.03, 0.08, 0.09],  # target seed 1
            [0.01, 0.0,  0.02, 0.08, 0.09],  # target seed 2  
            [0.03, 0.02, 0.0,  0.07, 0.08],  # close to both seeds
            [0.08, 0.08, 0.07, 0.0,  0.01],  # distant cluster
            [0.09, 0.09, 0.08, 0.01, 0.0]   # distant cluster
        ])
        
        target_indices = [0, 1]  # Both sequences 0 and 1 are seeds
        
        sequences = self._generate_test_sequences(len(distance_matrix))
        target_cluster, remaining, metrics = self.clustering.cluster(distance_matrix, target_indices, sequences)
        
        # Both seed sequences should be in target cluster
        assert 0 in target_cluster
        assert 1 in target_cluster
        
        # Sequence 2 should likely be merged (close to seeds)
        assert 2 in target_cluster
        
        # Distant sequences should remain separate
        assert 3 in remaining
        assert 4 in remaining
        
        assert len(target_cluster) == 3
        assert len(remaining) == 2
    
    def test_no_merging_due_to_threshold(self):
        """Test case where no sequences can be merged due to max_lump threshold."""
        # All inter-sequence distances exceed max_lump
        distance_matrix = np.array([
            [0.0,  0.10, 0.12, 0.15],  # target
            [0.10, 0.0,  0.11, 0.13],  
            [0.12, 0.11, 0.0,  0.14],
            [0.15, 0.13, 0.14, 0.0]
        ])
        
        target_indices = [0]
        
        sequences = self._generate_test_sequences(len(distance_matrix))
        target_cluster, remaining, metrics = self.clustering.cluster(distance_matrix, target_indices, sequences)
        
        # Only target sequence should be in cluster (no merging)
        assert target_cluster == [0]
        assert set(remaining) == {1, 2, 3}
        
        # Gap history should be empty (no merges performed)
        assert len(metrics['gap_history']) == 0
    
    def test_all_sequences_are_targets(self):
        """Test edge case where all sequences are targets."""
        distance_matrix = np.array([
            [0.0,  0.02, 0.03],
            [0.02, 0.0,  0.04],
            [0.03, 0.04, 0.0]
        ])
        
        target_indices = [0, 1, 2]  # All sequences are targets
        
        sequences = self._generate_test_sequences(len(distance_matrix))
        target_cluster, remaining, metrics = self.clustering.cluster(distance_matrix, target_indices, sequences)
        
        # All sequences should be in target cluster
        assert set(target_cluster) == {0, 1, 2}
        assert remaining == []
    
    def test_invalid_target_indices(self):
        """Test error handling for invalid target indices."""
        distance_matrix = np.array([
            [0.0,  0.02],
            [0.02, 0.0]
        ])
        
        # Test out of range index
        with pytest.raises(ValueError, match="Target index .* is out of range"):
            sequences = self._generate_test_sequences(len(distance_matrix))
            self.clustering.cluster(distance_matrix, [5], sequences)
        
        # Test negative index
        with pytest.raises(ValueError, match="Target index .* is out of range"):
            sequences = self._generate_test_sequences(len(distance_matrix))
            self.clustering.cluster(distance_matrix, [-1], sequences)
    
    def test_find_closest_to_target(self):
        """Test _find_closest_to_target helper method."""
        from gaphack.lazy_distances import DistanceProviderFactory
        
        distance_matrix = np.array([
            [0.0,  0.02, 0.04, 0.06],
            [0.02, 0.0,  0.07, 0.08],
            [0.04, 0.07, 0.0,  0.09],
            [0.06, 0.08, 0.09, 0.0]
        ])
        
        # Create distance provider from matrix
        distance_provider = DistanceProviderFactory.create_precomputed_provider(distance_matrix)
        
        target_cluster = {0}
        remaining = {1, 2, 3}
        
        closest_seq, distance = self.clustering._find_closest_to_target(
            target_cluster, remaining, distance_provider
        )
        
        # Sequence 1 is closest to target (distance 0.02)
        assert closest_seq == 1
        assert distance == 0.02
        
        # Test with multiple sequences in target cluster
        target_cluster = {0, 1}
        remaining = {2, 3}
        
        closest_seq, distance = self.clustering._find_closest_to_target(
            target_cluster, remaining, distance_provider
        )
        
        # Uses 95th percentile linkage (hardcoded in implementation):
        # sequence 2 has distances [0.04, 0.07] -> p95 ≈ 0.0685
        # sequence 3 has distances [0.06, 0.08] -> p95 ≈ 0.077
        # So sequence 2 should be closest with distance ≈ 0.0685
        # NOTE: Implementation should use self.target_percentile instead of hardcoded 95
        assert closest_seq == 2
        assert abs(distance - 0.0685) < 0.001  # Allow small numerical tolerance
    
    def test_gap_metrics_calculation(self):
        """Test gap metrics calculation for target vs remaining."""
        # Simple case with clear gap between target cluster and remaining
        distance_matrix = np.array([
            [0.0,  0.01, 0.08, 0.09],  # target - close to seq 1, far from 2,3
            [0.01, 0.0,  0.08, 0.09],  # close to target, far from 2,3
            [0.08, 0.08, 0.0,  0.01],  # far from target, close to seq 3
            [0.09, 0.09, 0.01, 0.0]   # far from target, close to seq 2
        ])
        
        target_indices = [0]
        
        sequences = self._generate_test_sequences(len(distance_matrix))
        target_cluster, remaining, metrics = self.clustering.cluster(distance_matrix, target_indices, sequences)
        
        # Should have clear gap between target cluster (0,1) and remaining (2,3)
        best_config = metrics['best_config']
        assert best_config['gap_size'] > 0
        
        # Gap metrics should be calculated
        assert 'gap_metrics' in best_config
        gap_metrics = best_config['gap_metrics']
        assert f'p{self.clustering.target_percentile}' in gap_metrics
    
    @patch('gaphack.target_clustering.tqdm')
    def test_progress_bar_control(self, mock_tqdm):
        """Test that progress bar is controlled by show_progress parameter."""
        distance_matrix = np.array([
            [0.0,  0.02],
            [0.02, 0.0]
        ])
        
        # Test with progress bar disabled
        clustering_no_progress = TargetModeClustering(show_progress=False)
        sequences = self._generate_test_sequences(len(distance_matrix))
        clustering_no_progress.cluster(distance_matrix, [0], sequences)
        mock_tqdm.assert_not_called()
        
        # Test with progress bar enabled  
        clustering_with_progress = TargetModeClustering(show_progress=True)
        sequences = self._generate_test_sequences(len(distance_matrix))
        clustering_with_progress.cluster(distance_matrix, [0], sequences)
        mock_tqdm.assert_called()
    
    def test_gap_history_tracking(self):
        """Test that gap history properly tracks clustering steps."""
        distance_matrix = np.array([
            [0.0,  0.01, 0.02, 0.03],
            [0.01, 0.0,  0.04, 0.05],
            [0.02, 0.04, 0.0,  0.06],
            [0.03, 0.05, 0.06, 0.0]
        ])
        
        target_indices = [0]
        
        sequences = self._generate_test_sequences(len(distance_matrix))
        target_cluster, remaining, metrics = self.clustering.cluster(distance_matrix, target_indices, sequences)
        
        gap_history = metrics['gap_history']
        
        # Should have history entries for each merge
        assert len(gap_history) > 0
        
        # Each history entry should have required fields
        for entry in gap_history:
            assert 'step' in entry
            assert 'target_cluster_size' in entry
            assert 'remaining_count' in entry
            assert 'merge_distance' in entry
            assert 'gap_size' in entry
            assert 'gap_exists' in entry
            assert 'merged_sequence' in entry
        
        # Steps should be sequential
        for i, entry in enumerate(gap_history):
            assert entry['step'] == i
        
        # Target cluster size should increase with each step
        sizes = [entry['target_cluster_size'] for entry in gap_history]
        for i in range(1, len(sizes)):
            assert sizes[i] == sizes[i-1] + 1
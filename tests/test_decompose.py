"""Tests for decomposition clustering functionality."""

import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from gaphack.decompose import (
    DecomposeClustering, 
    AssignmentTracker, 
    SupervisedTargetSelector,
    DecomposeResults
)
from gaphack.blast_neighborhood import BlastNeighborhoodFinder, SequenceCandidate


class TestAssignmentTracker:
    
    def test_init(self):
        """Test AssignmentTracker initialization."""
        tracker = AssignmentTracker()
        assert len(tracker.assignments) == 0
        assert len(tracker.assigned_sequences) == 0
    
    def test_assign_sequence(self):
        """Test single sequence assignment."""
        tracker = AssignmentTracker()
        
        tracker.assign_sequence("seq1", "cluster_001", 1)
        
        assert tracker.is_assigned("seq1")
        assert not tracker.is_assigned("seq2")
        assert "seq1" in tracker.assignments
        assert tracker.assignments["seq1"] == [("cluster_001", 1)]
    
    def test_assign_sequences(self):
        """Test multiple sequence assignment."""
        tracker = AssignmentTracker()
        
        tracker.assign_sequences(["seq1", "seq2"], "cluster_001", 1)
        
        assert tracker.is_assigned("seq1")
        assert tracker.is_assigned("seq2")
        assert len(tracker.assigned_sequences) == 2
    
    def test_conflicts_detection(self):
        """Test detection of multi-cluster assignments."""
        tracker = AssignmentTracker()
        
        # Assign seq1 to two clusters
        tracker.assign_sequence("seq1", "cluster_001", 1)
        tracker.assign_sequence("seq1", "cluster_002", 2)
        
        # Assign seq2 to one cluster
        tracker.assign_sequence("seq2", "cluster_001", 1)
        
        conflicts = tracker.get_conflicts()
        assert "seq1" in conflicts
        assert "seq2" not in conflicts
        assert set(conflicts["seq1"]) == {"cluster_001", "cluster_002"}
    
    def test_single_assignments(self):
        """Test getting sequences with single assignments."""
        tracker = AssignmentTracker()
        
        tracker.assign_sequence("seq1", "cluster_001", 1)
        tracker.assign_sequence("seq2", "cluster_001", 1)
        tracker.assign_sequence("seq3", "cluster_002", 2)
        
        single_assignments = tracker.get_single_assignments()
        assert len(single_assignments) == 3
        assert single_assignments["seq1"] == "cluster_001"
        assert single_assignments["seq2"] == "cluster_001"
        assert single_assignments["seq3"] == "cluster_002"
    
    def test_unassigned_sequences(self):
        """Test getting unassigned sequences."""
        tracker = AssignmentTracker()
        
        all_sequences = ["seq1", "seq2", "seq3", "seq4"]
        tracker.assign_sequence("seq1", "cluster_001", 1)
        tracker.assign_sequence("seq3", "cluster_001", 1)
        
        unassigned = tracker.get_unassigned(all_sequences)
        assert set(unassigned) == {"seq2", "seq4"}


class TestSupervisedTargetSelector:
    
    def test_init(self):
        """Test SupervisedTargetSelector initialization."""
        targets = ["target1", "target2", "target3"]
        selector = SupervisedTargetSelector(targets)
        assert selector.target_headers == targets
        assert len(selector.used_targets) == 0
    
    def test_get_next_target(self):
        """Test getting next target sequence."""
        targets = ["target1", "target2"]
        selector = SupervisedTargetSelector(targets)
        tracker = AssignmentTracker()
        
        # First call should return target1
        next_target = selector.get_next_target(tracker)
        assert next_target == ["target1"]
        assert "target1" in selector.used_targets
        
        # Second call should return target2
        next_target = selector.get_next_target(tracker)
        assert next_target == ["target2"]
        assert "target2" in selector.used_targets
        
        # Third call should return None
        next_target = selector.get_next_target(tracker)
        assert next_target is None
    
    def test_skip_assigned_targets(self):
        """Test that assigned targets are skipped."""
        targets = ["target1", "target2", "target3"]
        selector = SupervisedTargetSelector(targets)
        tracker = AssignmentTracker()
        
        # Assign target2 to a cluster
        tracker.assign_sequence("target2", "cluster_001", 1)
        
        # First call should return target1
        next_target = selector.get_next_target(tracker)
        assert next_target == ["target1"]
        
        # Second call should return target3 (skipping target2)
        next_target = selector.get_next_target(tracker)
        assert next_target == ["target3"]
    
    def test_has_more_targets(self):
        """Test checking for remaining targets."""
        targets = ["target1", "target2"]
        selector = SupervisedTargetSelector(targets)
        tracker = AssignmentTracker()
        
        assert selector.has_more_targets(tracker)
        
        # Get first target
        selector.get_next_target(tracker)
        assert selector.has_more_targets(tracker)
        
        # Get second target
        selector.get_next_target(tracker)
        assert not selector.has_more_targets(tracker)


class TestBlastNeighborhoodFinder:
    
    def test_hash_sequence(self):
        """Test sequence hashing."""
        sequences = ["ATCG", "GCTA"] 
        headers = ["seq1", "seq2"]
        finder = BlastNeighborhoodFinder(sequences, headers)
        
        hash1 = finder._hash_sequence("ATCG")
        hash2 = finder._hash_sequence("atcg")  # Different case
        hash3 = finder._hash_sequence("GCTA")
        
        assert hash1 == hash2  # Case insensitive
        assert hash1 != hash3  # Different sequences
        assert len(hash1) == 16  # Expected hash length
    
    def test_init_validation(self):
        """Test initialization validation."""
        with pytest.raises(ValueError, match="Number of sequences and headers must match"):
            BlastNeighborhoodFinder(["ATCG"], ["seq1", "seq2"])
    
    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch('gaphack.blast_neighborhood.Path.exists')
    def test_create_database(self, mock_exists, mock_run):
        """Test BLAST database creation."""
        sequences = ["ATCG", "GCTA"]
        headers = ["seq1", "seq2"]
        
        # Mock that database doesn't exist
        mock_exists.return_value = False
        
        # Mock successful makeblastdb
        mock_run.return_value = MagicMock(stdout="Database created successfully")
        
        finder = BlastNeighborhoodFinder(sequences, headers)
        
        # Verify makeblastdb was called
        mock_run.assert_called()
        args = mock_run.call_args[0][0]  # Get the command arguments
        assert 'makeblastdb' in args
        assert '-dbtype' in args
        assert 'nucl' in args


class TestSequenceCandidate:
    
    def test_sequence_candidate_creation(self):
        """Test SequenceCandidate dataclass."""
        candidate = SequenceCandidate(
            sequence_id="query1",
            sequence_hash="abc123",
            blast_identity=95.5,
            alignment_length=300,
            e_value=1e-10,
            bit_score=450.2
        )
        
        assert candidate.sequence_id == "query1"
        assert candidate.sequence_hash == "abc123"
        assert candidate.blast_identity == 95.5
        assert candidate.alignment_length == 300
        assert candidate.e_value == 1e-10
        assert candidate.bit_score == 450.2


class TestDecomposeResults:
    
    def test_decompose_results_creation(self):
        """Test DecomposeResults dataclass."""
        results = DecomposeResults()
        
        assert len(results.clusters) == 0
        assert len(results.unassigned) == 0
        assert len(results.conflicts) == 0
        assert len(results.iteration_summaries) == 0
        assert results.total_iterations == 0
        assert results.total_sequences_processed == 0
        assert results.coverage_percentage == 0.0
    
    def test_decompose_results_with_data(self):
        """Test DecomposeResults with actual data."""
        results = DecomposeResults(
            clusters={"cluster_001": ["seq1", "seq2"]},
            unassigned=["seq3"],
            total_iterations=2,
            total_sequences_processed=2,
            coverage_percentage=66.7
        )
        
        assert len(results.clusters) == 1
        assert "cluster_001" in results.clusters
        assert len(results.clusters["cluster_001"]) == 2
        assert len(results.unassigned) == 1
        assert results.total_iterations == 2
        assert results.total_sequences_processed == 2
        assert results.coverage_percentage == 66.7


class TestDecomposeClustering:
    
    def test_init(self):
        """Test DecomposeClustering initialization."""
        clustering = DecomposeClustering(
            min_split=0.01,
            max_lump=0.05,
            target_percentile=95,
            blast_max_hits=1000
        )
        
        assert clustering.min_split == 0.01
        assert clustering.max_lump == 0.05
        assert clustering.target_percentile == 95
        assert clustering.blast_max_hits == 1000
        assert clustering.target_clustering is not None
    
    @patch('gaphack.decompose.load_sequences_from_fasta')
    @patch('gaphack.decompose.BlastNeighborhoodFinder')
    def test_decompose_validation(self, mock_blast_finder, mock_load_sequences):
        """Test input validation in decompose method."""
        clustering = DecomposeClustering()
        
        # Test missing targets_fasta for supervised mode
        with pytest.raises(ValueError, match="targets_fasta is required for supervised mode"):
            clustering.decompose("input.fasta", strategy="supervised")
        
        # Test unsupported strategy
        with pytest.raises(NotImplementedError, match="Strategy 'unsupported' not yet implemented"):
            clustering.decompose("input.fasta", targets_fasta="targets.fasta", strategy="unsupported")
    
    @patch('gaphack.decompose.load_sequences_from_fasta')
    @patch('gaphack.decompose.BlastNeighborhoodFinder')
    @patch('gaphack.decompose.calculate_distance_matrix')
    def test_decompose_empty_targets(self, mock_calc_dist, mock_blast_finder, mock_load_sequences):
        """Test decompose with empty target list."""
        clustering = DecomposeClustering()
        
        # Mock input sequences
        mock_load_sequences.side_effect = [
            (["ATCG", "GCTA"], ["seq1", "seq2"]),  # input sequences
            ([], [])  # empty targets
        ]
        
        # Mock BLAST finder
        mock_finder_instance = MagicMock()
        mock_finder_instance.cleanup = MagicMock()
        mock_blast_finder.return_value = mock_finder_instance
        
        results = clustering.decompose("input.fasta", targets_fasta="empty_targets.fasta", strategy="supervised")
        
        # Should complete without error and return empty results
        assert len(results.clusters) == 0
        assert results.total_iterations == 0
        assert len(results.unassigned) == 2  # All sequences remain unassigned
        
        # Verify cleanup was called
        mock_finder_instance.cleanup.assert_called_once()
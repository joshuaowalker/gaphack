"""Tests for decomposition clustering functionality."""

import pytest
import tempfile
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from gaphack.decompose import (
    DecomposeClustering, 
    AssignmentTracker, 
    TargetSelector,
    NearbyTargetSelector,
    BlastResultMemory,
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


class TestTargetSelector:
    
    def test_init(self):
        """Test TargetSelector initialization."""
        targets = ["target1", "target2", "target3"]
        selector = TargetSelector(targets)
        assert selector.target_headers == targets
        assert len(selector.used_targets) == 0
    
    def test_get_next_target(self):
        """Test getting next target sequence."""
        targets = ["target1", "target2"]
        selector = TargetSelector(targets)
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
        selector = TargetSelector(targets)
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
        selector = TargetSelector(targets)
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
        with pytest.raises(ValueError, match="Unknown strategy 'unsupported'"):
            clustering.decompose("input.fasta", targets_fasta="targets.fasta", strategy="unsupported")
    
    @patch('gaphack.decompose.load_sequences_from_fasta')
    @patch('gaphack.decompose.BlastNeighborhoodFinder')
    def test_decompose_empty_targets(self, mock_blast_finder, mock_load_sequences):
        """Test decompose with empty target list."""
        clustering = DecomposeClustering()
        
        # Mock input sequences
        mock_load_sequences.side_effect = [
            (["ATCG", "GCTA"], ["seq1", "seq2"], {"seq1": "seq1", "seq2": "seq2"}),  # input sequences
            ([], [], {})  # empty targets
        ]
        
        # Mock BLAST finder
        mock_finder_instance = MagicMock()
        mock_finder_instance.cleanup = MagicMock()
        mock_blast_finder.return_value = mock_finder_instance
        
        # Should raise error when no matching targets found
        with pytest.raises(ValueError, match="No target sequences found matching sequences in input file"):
            clustering.decompose("input.fasta", targets_fasta="empty_targets.fasta", strategy="supervised")


class TestBlastResultMemory:
    
    def test_init(self):
        """Test BlastResultMemory initialization."""
        memory = BlastResultMemory()
        assert len(memory.unprocessed_neighborhoods) == 0
        assert len(memory.candidate_pool) == 0
        assert len(memory.fully_processed_targets) == 0
    
    def test_add_neighborhood(self):
        """Test adding BLAST neighborhoods."""
        memory = BlastResultMemory()
        
        memory.add_neighborhood("target1", ["seq1", "seq2", "seq3"])
        
        assert "target1" in memory.unprocessed_neighborhoods
        assert memory.unprocessed_neighborhoods["target1"] == {"seq1", "seq2", "seq3"}
        assert memory.candidate_pool == {"seq1", "seq2", "seq3"}
    
    def test_get_nearby_candidates(self):
        """Test getting nearby candidates."""
        memory = BlastResultMemory()
        tracker = AssignmentTracker()
        
        # Add neighborhood
        memory.add_neighborhood("target1", ["seq1", "seq2", "seq3"])
        
        # Initially all are candidates
        candidates = memory.get_nearby_candidates(tracker)
        assert set(candidates) == {"seq1", "seq2", "seq3"}
        
        # Assign one sequence
        tracker.assign_sequence("seq2", "cluster_001", 1)
        
        # Should return only unassigned
        candidates = memory.get_nearby_candidates(tracker)
        assert set(candidates) == {"seq1", "seq3"}
    
    def test_mark_processed(self):
        """Test marking sequences as processed."""
        memory = BlastResultMemory()
        
        # Add neighborhoods
        memory.add_neighborhood("target1", ["seq1", "seq2", "seq3"])
        memory.add_neighborhood("target2", ["seq3", "seq4", "seq5"])
        
        # Test overlap mode (default): sequences stay in pool
        memory.mark_processed(["seq2", "seq3"], allow_overlaps=True)
        
        # Check that sequences remain in pool
        assert "seq2" in memory.candidate_pool
        assert "seq3" in memory.candidate_pool
        assert memory.candidate_pool == {"seq1", "seq2", "seq3", "seq4", "seq5"}
        
        # Test no-overlap mode: sequences are removed from pool
        memory.mark_processed(["seq2", "seq3"], allow_overlaps=False)
        
        # Check that sequences are removed from pool
        assert "seq2" not in memory.candidate_pool
        assert "seq3" not in memory.candidate_pool
        assert memory.candidate_pool == {"seq1", "seq4", "seq5"}
        
        # Check that neighborhoods are updated
        assert memory.unprocessed_neighborhoods["target1"] == {"seq1"}
        assert memory.unprocessed_neighborhoods["target2"] == {"seq4", "seq5"}


class TestNearbyTargetSelector:
    
    def test_init(self):
        """Test NearbyTargetSelector initialization."""
        headers = ["seq1", "seq2", "seq3"]
        selector = NearbyTargetSelector(headers, max_clusters=5, max_sequences=100)
        
        assert selector.all_headers == headers
        assert selector.max_clusters == 5
        assert selector.max_sequences == 100
        assert selector.iteration_count == 0
        assert len(selector.used_targets) == 0
    
    def test_random_fallback(self):
        """Test random target selection when no BLAST memory."""
        headers = ["seq1", "seq2", "seq3"]
        selector = NearbyTargetSelector(headers, max_clusters=2)
        tracker = AssignmentTracker()
        
        # First call should return a target (random fallback)
        next_target = selector.get_next_target(tracker)
        assert next_target is not None
        assert len(next_target) == 1
        assert next_target[0] in headers
        assert selector.iteration_count == 1
        
        # Target should be marked as used
        assert next_target[0] in selector.used_targets
    
    def test_nearby_selection_with_blast_memory(self):
        """Test nearby selection using BLAST memory."""
        headers = ["seq1", "seq2", "seq3", "seq4", "seq5"]
        selector = SpiralTargetSelector(headers, max_clusters=3)
        tracker = AssignmentTracker()
        
        # Add BLAST neighborhood memory
        selector.add_blast_neighborhood("target1", ["seq2", "seq3", "seq4"])
        
        # Get next target - should prefer from BLAST neighborhood
        next_target = selector.get_next_target(tracker)
        assert next_target is not None
        assert next_target[0] in ["seq2", "seq3", "seq4"]
        
        # Mark sequence as processed
        selector.mark_sequences_processed([next_target[0]])
        
        # Should still get candidates from remaining neighborhood
        next_target_2 = selector.get_next_target(tracker)
        assert next_target_2 is not None
        assert next_target_2[0] != next_target[0]  # Different target
    
    def test_stopping_criteria(self):
        """Test stopping criteria for unsupervised mode."""
        headers = ["seq1", "seq2", "seq3"]
        
        # Test max_clusters limit
        selector = NearbyTargetSelector(headers, max_clusters=2)
        tracker = AssignmentTracker()
        
        assert selector.has_more_targets(tracker)
        
        # Simulate reaching max clusters
        selector.iteration_count = 2
        assert not selector.has_more_targets(tracker)
        
        # Test max_sequences limit with larger header set
        headers_large = ["seq1", "seq2", "seq3", "seq4", "seq5", "seq6", "seq7"]
        selector2 = NearbyTargetSelector(headers_large, max_sequences=5)
        tracker2 = AssignmentTracker()
        
        assert selector2.has_more_targets(tracker2)
        
        # Assign sequences up to limit
        tracker2.assign_sequence("seq1", "cluster_001", 1)
        tracker2.assign_sequence("seq2", "cluster_001", 1)
        tracker2.assign_sequence("seq3", "cluster_002", 2)
        
        # Still has targets (not at limit yet)
        assert selector2.has_more_targets(tracker2)
        
        # Add more sequences to reach limit
        tracker2.assign_sequence("seq4", "cluster_003", 3)
        tracker2.assign_sequence("seq5", "cluster_003", 3)
        
        # Should stop now (reached max_sequences)
        assert not selector2.has_more_targets(tracker2)


class TestDecomposeClusteringUnsupervised:
    
    def test_unsupervised_validation(self):
        """Test unsupervised mode input validation."""
        clustering = DecomposeClustering()
        
        # Test invalid strategy
        with pytest.raises(ValueError, match="Unknown strategy"):
            clustering.decompose("input.fasta", strategy="invalid")
    
    @patch('gaphack.decompose.load_sequences_from_fasta')
    @patch('gaphack.decompose.BlastNeighborhoodFinder')
    def test_unsupervised_exhaustive_mode(self, mock_blast_finder, mock_load_sequences):
        """Test unsupervised mode without stopping criteria (exhaustive)."""
        clustering = DecomposeClustering()
        
        # Mock sequences
        mock_load_sequences.return_value = (["ATCG", "GCTA"], ["seq1", "seq2"], {"seq1": "seq1", "seq2": "seq2"})
        
        # Mock BLAST finder
        mock_finder_instance = MagicMock()
        mock_finder_instance.cleanup = MagicMock()
        mock_blast_finder.return_value = mock_finder_instance
        
        # Should not raise error when no stopping criteria provided
        try:
            # This would normally run but we'll catch it before it actually executes BLAST
            clustering.decompose("input.fasta", strategy="unsupervised")
        except Exception as e:
            # Allow BLAST-related errors but not validation errors
            if "max_clusters" in str(e) or "max_sequences" in str(e):
                pytest.fail(f"Unexpected validation error: {e}")
            # Other errors (like BLAST setup) are expected in this mock test
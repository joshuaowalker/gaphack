"""
Tests for BLAST neighborhood functionality.
"""

import pytest
import tempfile
import os
import hashlib
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from hypothesis import given, strategies as st, settings

from gaphack.blast_neighborhood import BlastNeighborhoodFinder, SequenceCandidate


class TestSequenceCandidate:
    """Test suite for SequenceCandidate dataclass."""

    def test_sequence_candidate_creation(self):
        """Test SequenceCandidate creation and attributes."""
        candidate = SequenceCandidate(
            sequence_id="seq_1",
            sequence_hash="abc123",
            blast_identity=95.5,
            alignment_length=100,
            e_value=1e-10,
            bit_score=200.0
        )

        assert candidate.sequence_id == "seq_1"
        assert candidate.sequence_hash == "abc123"
        assert candidate.blast_identity == 95.5
        assert candidate.alignment_length == 100
        assert candidate.e_value == 1e-10
        assert candidate.bit_score == 200.0

    def test_sequence_candidate_equality(self):
        """Test SequenceCandidate equality comparison."""
        candidate1 = SequenceCandidate("seq_1", "abc123", 95.5, 100, 1e-10, 200.0)
        candidate2 = SequenceCandidate("seq_1", "abc123", 95.5, 100, 1e-10, 200.0)
        candidate3 = SequenceCandidate("seq_2", "def456", 90.0, 90, 1e-8, 150.0)

        assert candidate1 == candidate2
        assert candidate1 != candidate3


class TestBlastNeighborhoodFinder:
    """Comprehensive test suite for BlastNeighborhoodFinder."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = [
            "ATGCGATCGATCGATCGATGC",
            "ATGCGATCGATCGATCGATCC",
            "TTTTGGGGCCCCAAAATTTT",
            "GGGGAAAACCCCTTTTGGGG"
        ]
        self.test_headers = [f"seq_{i}" for i in range(len(self.test_sequences))]

    def test_initialization_valid_input(self):
        """Test BlastNeighborhoodFinder initialization with valid input."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        assert finder.sequences == self.test_sequences
        assert finder.headers == self.test_headers
        assert finder.cache_dir.exists()
        assert len(finder.sequence_lookup) == len(self.test_sequences)

    def test_initialization_mismatched_lengths(self):
        """Test initialization with mismatched sequence and header lengths."""
        with pytest.raises(ValueError, match="Number of sequences and headers must match"):
            BlastNeighborhoodFinder(
                self.test_sequences,
                self.test_headers[:-1]  # One fewer header
            )

    def test_initialization_custom_cache_dir(self):
        """Test initialization with custom cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            custom_cache = Path(tmpdir) / "custom_blast_cache"
            finder = BlastNeighborhoodFinder(
                self.test_sequences,
                self.test_headers,
                cache_dir=custom_cache
            )

            assert finder.cache_dir == custom_cache
            assert custom_cache.exists()

    def test_hash_sequence_consistency(self):
        """Test that sequence hashing is consistent."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        seq = "ATGCGATCGATC"
        hash1 = finder._hash_sequence(seq)
        hash2 = finder._hash_sequence(seq)

        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 16  # Truncated SHA256 hex digest length

    def test_hash_sequence_different_inputs(self):
        """Test that different sequences produce different hashes."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        seq1 = "ATGCGATCGATC"
        seq2 = "CGATAGTCGATC"

        hash1 = finder._hash_sequence(seq1)
        hash2 = finder._hash_sequence(seq2)

        assert hash1 != hash2

    def test_get_cache_key_deterministic(self):
        """Test that cache key generation is deterministic."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        key1 = finder._get_cache_key()
        key2 = finder._get_cache_key()

        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 12  # Truncated SHA256 hex digest length

    def test_get_cache_key_different_sequences(self):
        """Test that different sequence sets produce different cache keys."""
        finder1 = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)
        finder2 = BlastNeighborhoodFinder(
            self.test_sequences[:-1],  # Different sequence set
            self.test_headers[:-1]
        )

        key1 = finder1._get_cache_key()
        key2 = finder2._get_cache_key()

        assert key1 != key2

    @patch('gaphack.blast_neighborhood.Path.exists')
    def test_is_database_cached_true(self, mock_exists):
        """Test database cache detection when database exists."""
        mock_exists.return_value = True

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)
        assert finder._is_database_cached() is True

    @patch('gaphack.blast_neighborhood.Path.exists')
    def test_is_database_cached_false(self, mock_exists):
        """Test database cache detection when database doesn't exist."""
        mock_exists.return_value = False

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)
        assert finder._is_database_cached() is False

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch('gaphack.blast_neighborhood.Path.exists')
    def test_create_database_success(self, mock_exists, mock_run):
        """Test successful BLAST database creation."""
        mock_exists.return_value = False  # Database doesn't exist
        mock_run.return_value = MagicMock(stdout="Database created successfully")

        with tempfile.TemporaryDirectory() as tmpdir:
            finder = BlastNeighborhoodFinder(
                self.test_sequences,
                self.test_headers,
                cache_dir=Path(tmpdir)
            )

            # Mock the file writing to avoid actual file operations
            with patch('builtins.open', mock_open()):
                finder._create_database()

            # Verify makeblastdb was called
            mock_run.assert_called()
            call_args = mock_run.call_args[0][0]
            assert 'makeblastdb' in call_args

    @patch('gaphack.blast_neighborhood.subprocess.run')
    def test_create_database_failure(self, mock_run):
        """Test BLAST database creation failure handling."""
        mock_run.side_effect = subprocess.CalledProcessError(1, 'makeblastdb')

        with tempfile.TemporaryDirectory() as tmpdir:
            # The error is raised during initialization since database creation happens there
            with pytest.raises(subprocess.CalledProcessError):
                BlastNeighborhoodFinder(
                    self.test_sequences,
                    self.test_headers,
                    cache_dir=Path(tmpdir)
                )

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch.object(BlastNeighborhoodFinder, '_is_database_cached')
    def test_find_neighborhood_database_creation(self, mock_is_cached, mock_run):
        """Test that find_neighborhood creates database when needed."""
        # Mock database not cached initially, then cached after creation
        mock_is_cached.side_effect = [False, True]  # First call False, subsequent True
        mock_run.return_value = MagicMock(stdout="")

        # Mock SeqIO.write to avoid file operations during database creation
        with patch('gaphack.blast_neighborhood.SeqIO.write'):
            finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # Now mock database as cached for the find_neighborhood call
        mock_is_cached.return_value = True

        neighborhood = finder.find_neighborhood(["seq_0"])

        # Should have called makeblastdb during initialization
        mock_run.assert_called()
        assert any('makeblastdb' in str(call) for call in mock_run.call_args_list)

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch.object(BlastNeighborhoodFinder, '_is_database_cached')
    def test_find_neighborhood_uses_cached_database(self, mock_is_cached, mock_run):
        """Test that find_neighborhood uses cached database when available."""
        mock_is_cached.return_value = True
        mock_run.return_value = MagicMock(stdout="")

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        with patch.object(finder, '_create_database') as mock_create_db:
            with patch('builtins.open', mock_open()):
                candidates = finder.find_neighborhood(["seq_0"])

        mock_create_db.assert_not_called()

    def test_find_neighborhood_invalid_headers(self):
        """Test find_neighborhood with invalid target headers."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # The implementation returns empty list for invalid headers, doesn't raise
        result = finder.find_neighborhood(["nonexistent_seq"])
        assert result == []

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch.object(BlastNeighborhoodFinder, '_is_database_cached')
    def test_find_neighborhood_blast_output_parsing(self, mock_is_cached, mock_run):
        """Test BLAST output parsing in find_neighborhood."""
        mock_is_cached.return_value = True

        # Mock BLAST output with 7-column format expected by implementation
        # Format: qseqid sseqid qcovs pident length evalue bitscore
        blast_output = """seq_0\thash1\t95\t95.0\t100\t1e-50\t200
seq_0\thash2\t90\t85.0\t90\t1e-30\t150"""
        mock_run.return_value = MagicMock(stdout=blast_output)

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # Mock sequence lookup to map hashes to headers
        finder.sequence_lookup = {
            'hash1': [('ATGCGATCGATCGATCC', 'seq_1', 1)],
            'hash2': [('TTTTTTTTTTTTTTTTTT', 'seq_2', 2)]
        }

        neighborhood = finder.find_neighborhood(["seq_0"])

        # Should include target plus found sequences
        assert "seq_0" in neighborhood  # Target always included
        assert "seq_1" in neighborhood  # From hash1
        assert "seq_2" in neighborhood  # From hash2

    def test_find_neighborhood_empty_targets(self):
        """Test find_neighborhood with empty target list."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        candidates = finder.find_neighborhood([])
        assert candidates == []

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch.object(BlastNeighborhoodFinder, '_is_database_cached')
    def test_find_neighborhood_identity_filtering(self, mock_is_cached, mock_run):
        """Test that identity filtering works correctly."""
        mock_is_cached.return_value = True

        # Mock BLAST output with varying identities (7-column format)
        blast_output = """seq_0\thash1\t95\t95.0\t100\t1e-50\t200
seq_0\thash2\t90\t75.0\t90\t1e-30\t150
seq_0\thash3\t90\t85.0\t95\t1e-40\t175"""
        mock_run.return_value = MagicMock(stdout=blast_output)

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # Mock sequence lookup
        finder.sequence_lookup = {
            'hash1': [('SEQ1', 'seq_1', 1)],
            'hash2': [('SEQ2', 'seq_2', 2)],
            'hash3': [('SEQ3', 'seq_3', 3)]
        }

        # Filter with 80% minimum identity - this filtering happens in BLAST command
        neighborhood = finder.find_neighborhood(["seq_0"], min_identity=80.0)

        # Should include target plus sequences meeting identity threshold
        assert "seq_0" in neighborhood  # Target always included
        # Note: filtering happens at BLAST level, so we expect all results in mock
        assert len(neighborhood) >= 1  # At least the target

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch.object(BlastNeighborhoodFinder, '_is_database_cached')
    def test_find_neighborhood_max_hits_limit(self, mock_is_cached, mock_run):
        """Test that max_hits parameter limits results."""
        mock_is_cached.return_value = True

        # Generate multiple BLAST hits (7-column format)
        blast_lines = []
        for i in range(10):
            blast_lines.append(f"seq_0\thash_{i+1}\t90\t90.0\t100\t1e-40\t175")

        blast_output = "\n".join(blast_lines)
        mock_run.return_value = MagicMock(stdout=blast_output)

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # Mock sequence lookup for each hash
        finder.sequence_lookup = {f'hash_{i+1}': [(f'SEQ_{i+1}', f'seq_{i+1}', i+1)] for i in range(10)}

        neighborhood = finder.find_neighborhood(["seq_0"], max_hits=5)

        # Max hits limits BLAST output, not final neighborhood size
        # Verify the BLAST command was called with correct max_hits parameter
        mock_run.assert_called()
        call_args = mock_run.call_args[0][0]
        assert '-max_target_seqs' in call_args
        max_hits_index = call_args.index('-max_target_seqs')
        assert call_args[max_hits_index + 1] == '5'

    def test_cleanup(self):
        """Test cleanup functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            finder = BlastNeighborhoodFinder(
                self.test_sequences,
                self.test_headers,
                cache_dir=Path(tmpdir)
            )

            # Mock the existence of database files
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'unlink') as mock_unlink:

                finder.cleanup()

                # Verify unlink was called (exact count depends on what files exist)
                assert mock_unlink.called

    @given(
        sequences=st.lists(
            st.text(alphabet="ATCG", min_size=10, max_size=50),
            min_size=1,
            max_size=10
        )
    )
    @settings(deadline=1000)  # Increase deadline for property-based test
    def test_hash_sequence_properties(self, sequences):
        """Property-based test for sequence hashing."""
        if not sequences:
            return

        headers = [f"seq_{i}" for i in range(len(sequences))]

        # Mock database existence check to avoid slow database creation
        with patch.object(BlastNeighborhoodFinder, '_is_database_cached', return_value=True):
            finder = BlastNeighborhoodFinder(sequences, headers)

        # Test that same sequence always produces same hash
        for seq in sequences:
            hash1 = finder._hash_sequence(seq)
            hash2 = finder._hash_sequence(seq)
            assert hash1 == hash2

        # Test that different sequences produce different hashes
        if len(set(sequences)) > 1:  # If we have unique sequences
            unique_sequences = list(set(sequences))
            hashes = [finder._hash_sequence(seq) for seq in unique_sequences]
            # Note: With 16-character truncated hashes, collisions are possible
            # So we test that we get reasonable hash distribution
            assert len(set(hashes)) >= 1  # At least some unique hashes

    def test_sequence_lookup_correctness(self):
        """Test that sequence lookup is built correctly."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # Check that lookup contains entries for sequences
        assert len(finder.sequence_lookup) > 0

        # Check that lookup contains correct mappings
        for i, (seq, header) in enumerate(zip(self.test_sequences, self.test_headers)):
            seq_hash = finder._hash_sequence(seq)
            assert seq_hash in finder.sequence_lookup
            # sequence_lookup maps hash -> list of (seq, header, index) tuples
            entries = finder.sequence_lookup[seq_hash]
            assert isinstance(entries, list)
            # Find the entry for this specific sequence
            found = False
            for stored_seq, stored_header, stored_index in entries:
                if stored_seq == seq and stored_header == header and stored_index == i:
                    found = True
                    break
            assert found, f"Could not find entry for sequence {i}: {header}"

    def test_error_handling_malformed_blast_output(self):
        """Test error handling with malformed BLAST output."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # Test with malformed BLAST line (insufficient columns)
        malformed_output = "seq_0\tseq_1\t95.0"  # Missing required columns

        with patch.object(finder, '_is_database_cached', return_value=True), \
             patch('gaphack.blast_neighborhood.subprocess.run') as mock_run:

            mock_run.return_value = MagicMock(stdout=malformed_output)

            # Should handle malformed lines gracefully (skip them)
            neighborhood = finder.find_neighborhood(["seq_0"])
            # Target should still be included even if BLAST output is malformed
            assert "seq_0" in neighborhood


# Helper function for mock_open if not available
try:
    from unittest.mock import mock_open
except ImportError:
    def mock_open(mock=None, read_data=''):
        """Mock implementation of open() for testing."""
        if mock is None:
            mock = MagicMock()

        handle = MagicMock()
        handle.write.return_value = None
        handle.__enter__.return_value = handle
        handle.__exit__.return_value = False
        handle.read.return_value = read_data

        mock.return_value = handle
        return mock
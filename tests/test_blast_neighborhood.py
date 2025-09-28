"""
Tests for BLAST neighborhood functionality.
"""

import pytest
import tempfile
import os
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from hypothesis import given, strategies as st

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
        assert len(hash1) == 64  # SHA256 hex digest length

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
        assert len(key1) == 64  # SHA256 hex digest length

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
            finder = BlastNeighborhoodFinder(
                self.test_sequences,
                self.test_headers,
                cache_dir=Path(tmpdir)
            )

            with patch('builtins.open', mock_open()):
                with pytest.raises(RuntimeError, match="Failed to create BLAST database"):
                    finder._create_database()

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch.object(BlastNeighborhoodFinder, '_is_database_cached')
    @patch.object(BlastNeighborhoodFinder, '_create_database')
    def test_find_neighborhood_database_creation(self, mock_create_db, mock_is_cached, mock_run):
        """Test that find_neighborhood creates database when needed."""
        mock_is_cached.return_value = False
        mock_run.return_value = MagicMock(stdout="")

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        with patch('builtins.open', mock_open()):
            candidates = finder.find_neighborhood(["seq_0"])

        mock_create_db.assert_called_once()

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

        with pytest.raises(ValueError, match="Target header .* not found"):
            finder.find_neighborhood(["nonexistent_seq"])

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch.object(BlastNeighborhoodFinder, '_is_database_cached')
    def test_find_neighborhood_blast_output_parsing(self, mock_is_cached, mock_run):
        """Test BLAST output parsing in find_neighborhood."""
        mock_is_cached.return_value = True

        # Mock BLAST output with specific format
        blast_output = """# BLAST output
seq_0\tseq_1\t95.0\t100\t5\t0\t1\t100\t1\t100\t1e-50\t200
seq_0\tseq_2\t85.0\t90\t15\t0\t1\t90\t1\t90\t1e-30\t150
"""
        mock_run.return_value = MagicMock(stdout=blast_output)

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        with patch('builtins.open', mock_open()):
            candidates = finder.find_neighborhood(["seq_0"])

        # Should parse two BLAST hits
        assert len(candidates) == 2

        # Check first candidate
        candidate1 = candidates[0]
        assert candidate1.sequence_id == "seq_1"
        assert candidate1.blast_identity == 95.0
        assert candidate1.alignment_length == 100
        assert candidate1.e_value == 1e-50
        assert candidate1.bit_score == 200.0

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

        # Mock BLAST output with varying identities
        blast_output = """seq_0\tseq_1\t95.0\t100\t5\t0\t1\t100\t1\t100\t1e-50\t200
seq_0\tseq_2\t75.0\t90\t15\t0\t1\t90\t1\t90\t1e-30\t150
seq_0\tseq_3\t85.0\t95\t10\t0\t1\t95\t1\t95\t1e-40\t175
"""
        mock_run.return_value = MagicMock(stdout=blast_output)

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        with patch('builtins.open', mock_open()):
            # Filter with 80% minimum identity
            candidates = finder.find_neighborhood(["seq_0"], min_identity=80.0)

        # Should only include sequences with >= 80% identity
        assert len(candidates) == 2  # seq_1 (95%) and seq_3 (85%)

        sequence_ids = [c.sequence_id for c in candidates]
        assert "seq_1" in sequence_ids
        assert "seq_3" in sequence_ids
        assert "seq_2" not in sequence_ids  # 75% < 80%

    @patch('gaphack.blast_neighborhood.subprocess.run')
    @patch.object(BlastNeighborhoodFinder, '_is_database_cached')
    def test_find_neighborhood_max_hits_limit(self, mock_is_cached, mock_run):
        """Test that max_hits parameter limits results."""
        mock_is_cached.return_value = True

        # Generate multiple BLAST hits
        blast_lines = []
        for i in range(10):
            blast_lines.append(f"seq_0\tseq_{i+1}\t90.0\t100\t10\t0\t1\t100\t1\t100\t1e-40\t175")

        blast_output = "\n".join(blast_lines)
        mock_run.return_value = MagicMock(stdout=blast_output)

        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        with patch('builtins.open', mock_open()):
            candidates = finder.find_neighborhood(["seq_0"], max_hits=5)

        # Should be limited to 5 hits
        assert len(candidates) == 5

    @patch('os.remove')
    def test_cleanup(self, mock_remove):
        """Test cleanup functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            finder = BlastNeighborhoodFinder(
                self.test_sequences,
                self.test_headers,
                cache_dir=Path(tmpdir)
            )

            # Mock the existence of database files
            with patch.object(Path, 'exists', return_value=True), \
                 patch.object(Path, 'glob') as mock_glob:

                mock_glob.return_value = [
                    Path(tmpdir) / "db.nhr",
                    Path(tmpdir) / "db.nin",
                    Path(tmpdir) / "db.nsq"
                ]

                finder.cleanup()

                # Verify files were attempted to be removed
                assert mock_remove.call_count == 3

    @given(
        sequences=st.lists(
            st.text(alphabet="ATCG", min_size=10, max_size=50),
            min_size=1,
            max_size=10
        )
    )
    def test_hash_sequence_properties(self, sequences):
        """Property-based test for sequence hashing."""
        if not sequences:
            return

        headers = [f"seq_{i}" for i in range(len(sequences))]
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
            assert len(set(hashes)) == len(unique_sequences)

    def test_sequence_lookup_correctness(self):
        """Test that sequence lookup is built correctly."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # Check that all sequences are in lookup
        assert len(finder.sequence_lookup) == len(self.test_sequences)

        # Check that lookup contains correct mappings
        for i, (seq, header) in enumerate(zip(self.test_sequences, self.test_headers)):
            seq_hash = finder._hash_sequence(seq)
            assert seq_hash in finder.sequence_lookup
            stored_seq, stored_header = finder.sequence_lookup[seq_hash]
            assert stored_seq == seq
            assert stored_header == header

    def test_error_handling_malformed_blast_output(self):
        """Test error handling with malformed BLAST output."""
        finder = BlastNeighborhoodFinder(self.test_sequences, self.test_headers)

        # Test with malformed BLAST line (insufficient columns)
        malformed_output = "seq_0\tseq_1\t95.0"  # Missing required columns

        with patch.object(finder, '_is_database_cached', return_value=True), \
             patch('gaphack.blast_neighborhood.subprocess.run') as mock_run, \
             patch('builtins.open', mock_open()):

            mock_run.return_value = MagicMock(stdout=malformed_output)

            # Should handle malformed lines gracefully (skip them)
            candidates = finder.find_neighborhood(["seq_0"])
            assert len(candidates) == 0  # No valid candidates from malformed output


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
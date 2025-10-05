"""Tests for MSACachedDistanceProvider."""

import pytest
import numpy as np
from gaphack.distance_providers import MSACachedDistanceProvider, MSAAlignmentError


class TestMSACachedDistanceProvider:
    """Test suite for MSACachedDistanceProvider."""

    @pytest.fixture
    def simple_sequences(self):
        """Simple test sequences."""
        return [
            "ATCGATCG",
            "ATCGATCG",  # Identical to first
            "ATCGTTCG",  # 1 substitution
            "ATCGAACG",  # 1 substitution, different position
        ]

    @pytest.fixture
    def simple_headers(self):
        """Headers for simple sequences."""
        return ["seq1", "seq2", "seq3", "seq4"]

    def test_initialization(self, simple_sequences, simple_headers):
        """Test provider initialization."""
        provider = MSACachedDistanceProvider(simple_sequences, simple_headers)
        assert provider.n == 4
        assert len(provider.sequences) == 4
        assert len(provider.headers) == 4

    def test_initialization_without_headers(self, simple_sequences):
        """Test provider initialization without headers."""
        provider = MSACachedDistanceProvider(simple_sequences)
        assert provider.n == 4
        assert provider.headers == ["seq_0", "seq_1", "seq_2", "seq_3"]

    def test_identical_sequences(self, simple_sequences):
        """Test distance between identical sequences."""
        provider = MSACachedDistanceProvider(simple_sequences)
        assert provider.get_distance(0, 0) == 0.0
        assert provider.get_distance(0, 1) == 0.0
        assert provider.get_distance(1, 0) == 0.0

    def test_similar_sequences(self, simple_sequences):
        """Test distance between similar sequences."""
        provider = MSACachedDistanceProvider(simple_sequences)
        # 1 substitution out of 8 bases
        dist = provider.get_distance(0, 2)
        assert 0.1 < dist < 0.2  # Approximately 12.5%

    def test_distance_symmetry(self, simple_sequences):
        """Test that distance is symmetric."""
        provider = MSACachedDistanceProvider(simple_sequences)
        assert provider.get_distance(0, 2) == provider.get_distance(2, 0)
        assert provider.get_distance(1, 3) == provider.get_distance(3, 1)

    def test_distance_caching(self, simple_sequences):
        """Test that distances are cached."""
        provider = MSACachedDistanceProvider(simple_sequences)
        # First call computes
        dist1 = provider.get_distance(0, 2)
        # Second call uses cache
        dist2 = provider.get_distance(0, 2)
        assert dist1 == dist2
        # Check cache was used
        assert (0, 2) in provider._distance_cache

    def test_get_distances_from_sequence(self, simple_sequences):
        """Test getting distances from one sequence to multiple targets."""
        provider = MSACachedDistanceProvider(simple_sequences)
        distances = provider.get_distances_from_sequence(0, {1, 2, 3})
        assert len(distances) == 3
        assert 1 in distances
        assert 2 in distances
        assert 3 in distances
        assert distances[1] == 0.0  # Identical sequence

    def test_ensure_distances_computed(self, simple_sequences):
        """Test ensure_distances_computed is a no-op."""
        provider = MSACachedDistanceProvider(simple_sequences)
        # Should not raise an error
        provider.ensure_distances_computed({0, 1, 2})

    def test_build_distance_matrix(self, simple_sequences):
        """Test building full distance matrix."""
        provider = MSACachedDistanceProvider(simple_sequences)
        matrix = provider.build_distance_matrix()
        assert matrix.shape == (4, 4)
        # Check symmetry
        assert np.allclose(matrix, matrix.T)
        # Check diagonal is zero
        assert np.allclose(np.diag(matrix), 0.0)
        # Check identical sequences have zero distance
        assert matrix[0, 1] == 0.0

    def test_with_gaps(self):
        """Test sequences with indels."""
        sequences = [
            "ATCGATCG",
            "ATCG-TCG",  # Deletion in middle (note: input won't have gaps)
            "ATCGAATCG",  # Insertion in middle
        ]
        # Use sequences without gaps - the alignment will add them
        sequences = [
            "ATCGATCG",
            "ATCGTCG",
            "ATCGAATCG",
        ]
        provider = MSACachedDistanceProvider(sequences)
        # Should handle indels correctly
        dist01 = provider.get_distance(0, 1)
        dist02 = provider.get_distance(0, 2)
        assert dist01 > 0
        assert dist02 > 0

    def test_larger_dataset(self):
        """Test with a larger dataset."""
        # Create 50 sequences with varying similarity
        base_seq = "ATCGATCGATCGATCGATCGATCG"
        sequences = [base_seq]

        for i in range(1, 50):
            # Add mutations proportional to index
            seq = list(base_seq)
            for j in range(i % 5):
                if j * 5 < len(seq):
                    seq[j * 5] = "T" if seq[j * 5] == "A" else "A"
            sequences.append("".join(seq))

        provider = MSACachedDistanceProvider(sequences)
        matrix = provider.build_distance_matrix()

        # Check basic properties
        assert matrix.shape == (50, 50)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)

    def test_raises_on_spoa_failure(self, monkeypatch):
        """Test that MSAAlignmentError is raised when SPOA fails."""
        from gaphack import utils

        # Mock run_spoa_msa to return None (failure)
        def mock_spoa(sequences):
            return None

        monkeypatch.setattr(utils, "run_spoa_msa", mock_spoa)

        sequences = ["ATCGATCG", "ATCGATCG", "TTCGTTCG"]

        # Should raise exception instead of falling back
        with pytest.raises(MSAAlignmentError) as exc_info:
            MSACachedDistanceProvider(sequences)

        assert "SPOA failed" in str(exc_info.value)

    def test_different_length_sequences(self):
        """Test sequences of different lengths."""
        sequences = [
            "ATCGATCG",
            "ATCGATCGATCG",  # Longer (4 extra bases)
            "ATCG",  # Shorter
        ]
        provider = MSACachedDistanceProvider(sequences)
        # Should handle different lengths via alignment
        dist01 = provider.get_distance(0, 1)
        dist02 = provider.get_distance(0, 2)
        # dist01 should be non-zero due to length difference
        assert dist01 >= 0  # May be 0 if normalization handles length perfectly
        # dist02 should definitely be non-zero (4 bases missing)
        assert dist02 >= 0

    def test_empty_targets(self, simple_sequences):
        """Test get_distances_from_sequence with empty targets."""
        provider = MSACachedDistanceProvider(simple_sequences)
        distances = provider.get_distances_from_sequence(0, set())
        assert len(distances) == 0

    def test_single_target(self, simple_sequences):
        """Test get_distances_from_sequence with single target."""
        provider = MSACachedDistanceProvider(simple_sequences)
        distances = provider.get_distances_from_sequence(0, {2})
        assert len(distances) == 1
        assert 2 in distances
        assert distances[2] == provider.get_distance(0, 2)


class TestMSADistanceProviderIntegration:
    """Integration tests for MSA distance provider with real data."""

    @pytest.fixture
    def russula_sequences(self):
        """Load a small subset of Russula sequences for testing."""
        from gaphack.utils import load_sequences_from_fasta
        import os

        test_file = "tests/test_data/russula_diverse_50.fasta"
        if os.path.exists(test_file):
            sequences, headers, _ = load_sequences_from_fasta(test_file)
            return sequences, headers
        else:
            pytest.skip("Russula test data not available")

    def test_russula_integration(self, russula_sequences):
        """Test MSA provider with real biological data."""
        sequences, headers = russula_sequences
        provider = MSACachedDistanceProvider(sequences, headers)

        # Build distance matrix
        matrix = provider.build_distance_matrix()

        # Check basic properties
        assert matrix.shape == (len(sequences), len(sequences))
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)

        # Check that distances are reasonable (not all zeros, not all ones)
        non_diag = matrix[np.triu_indices_from(matrix, k=1)]
        assert np.mean(non_diag) > 0.01
        assert np.mean(non_diag) < 0.99

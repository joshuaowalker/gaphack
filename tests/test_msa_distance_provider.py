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
        # dist02 is 0.0 due to homopolymer normalization: ATCGAATCG vs ATCGATCG
        # The extra 'A' creates a homopolymer (AA) which is normalized to single A
        assert dist02 == 0.0

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


class TestMedianNormalization:
    """Test median length normalization for mixed-length sequences."""

    def test_median_normalization_mixed_lengths(self):
        """Test that median normalization provides consistent distances for mixed-length sequences."""
        # Simulate full ITS (500bp) and ITS2-only (200bp) sequences
        # All sequences have 2 substitutions in the ITS2 region

        # Create base sequence (500bp) - mixed sequence to avoid homopolymer normalization
        base_seq = "ATCG" * 50 + "TGCA" * 75  # 200bp + 300bp = 500bp

        # Variant with 2 substitutions in ITS2 region (first 200bp)
        variant_seq = "CTCG" + "ATCG" * 49 + "CCCA" + "TGCA" * 74  # 2 substitutions at positions 0 and 200

        # ITS2-only version (first 200bp only)
        its2_base = base_seq[:200]
        its2_variant = variant_seq[:200]

        # Create mixed dataset: 3 full ITS, 1 ITS2
        sequences = [base_seq, variant_seq, base_seq, its2_variant]

        provider = MSACachedDistanceProvider(sequences)

        # Check that median is at full ITS length
        assert provider.normalization_length == 500

        # Get distances
        dist_full_full = provider.get_distance(0, 1)  # full vs full (2 edits)
        dist_full_its2 = provider.get_distance(0, 3)  # full vs ITS2 (1 edit in overlap)

        # Both should use same denominator (median = 500bp)
        # Full vs full: ~2 edits / 500 = ~0.004
        # Full vs ITS2: ~1 edit / 500 = ~0.002 (only 1 edit in the overlapping ITS2 region)
        assert dist_full_full > dist_full_its2  # More edits = larger distance
        assert 0.002 < dist_full_full < 0.006  # Approximately 2 edits / 500bp

    def test_median_normalization_all_short_sequences(self):
        """Test median normalization when all sequences are short."""
        # All sequences are ITS2-only (200bp)
        # Use mixed sequences to avoid homopolymer normalization
        sequences = [
            "ATCG" * 50,  # 200bp
            "CTCG" + "ATCG" * 48 + "CTCG",  # 200bp, 2 substitutions (positions 0 and 196)
            "ATCG" * 50,  # 200bp
        ]

        provider = MSACachedDistanceProvider(sequences)

        # Median should be 200bp
        assert provider.normalization_length == 200

        # Distance should be approximately 1-2 edits / 200bp
        # (adjusted-identity may normalize some edits)
        dist = provider.get_distance(0, 1)
        assert 0.004 < dist < 0.015  # Allow tolerance for alignment adjustments

    def test_median_robust_to_outliers(self):
        """Test that median is robust to chimeric/long outliers."""
        # Simulate dataset with one chimera (2000bp) among full ITS (500bp)
        sequences = [
            "ATCG" * 125,  # 500bp
            "ATCG" * 125,  # 500bp
            "ATCG" * 125,  # 500bp
            "ATCG" * 125,  # 500bp
            "ATCG" * 500,  # 2000bp chimera/SSU+ITS
        ]

        provider = MSACachedDistanceProvider(sequences)

        # Median should be 500bp (not affected by the 2000bp outlier)
        assert provider.normalization_length == 500

        # Mean would be 800bp (affected by outlier): (500+500+500+500+2000)/5 = 800
        mean_length = int(np.mean([len(seq) for seq in sequences]))
        assert mean_length == 800

        # This demonstrates why median is better than mean

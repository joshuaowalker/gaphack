"""
Tests for utility functions.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from gaphack.utils import (
    load_sequences_from_fasta,
    calculate_distance_matrix,
    format_cluster_output,
    save_clusters_to_file,
    validate_sequences,
    compute_msa_distance,
    filter_msa_positions,
    replace_terminal_gaps,
    run_spoa_msa
)


class TestSequenceLoading:
    """Test sequence loading functions."""
    
    def test_load_sequences_from_fasta(self):
        """Test loading sequences from a FASTA file."""
        # Create a temporary FASTA file
        fasta_content = """>seq1
ATCGATCG
>seq2
GCTAGCTA
>seq3
TTTTAAAA
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(fasta_content)
            temp_path = f.name
        
        try:
            sequences, headers, header_mapping = load_sequences_from_fasta(temp_path)

            assert len(sequences) == 3
            assert len(headers) == 3
            assert len(header_mapping) == 3
            assert sequences[0] == "ATCGATCG"
            assert sequences[1] == "GCTAGCTA"
            assert sequences[2] == "TTTTAAAA"
            assert headers[0] == "seq1"
            assert headers[1] == "seq2"
            assert headers[2] == "seq3"
        finally:
            Path(temp_path).unlink()


class TestDistanceCalculations:
    """Test distance calculation functions."""
    
    def test_calculate_distance_matrix(self):
        """Test distance matrix calculation."""
        sequences = ["ATCG", "ATCC", "TACG", "CGTA"]
        # We can't test exact values without mocking adjusted-identity
        # So just test matrix properties
        distance_matrix = calculate_distance_matrix(sequences)
        
        # Check shape
        assert distance_matrix.shape == (4, 4)
        
        # Check diagonal is zero
        for i in range(4):
            assert distance_matrix[i, i] == 0.0
        
        # Check symmetry
        for i in range(4):
            for j in range(4):
                assert distance_matrix[i, j] == distance_matrix[j, i]
        
        # Check values are in valid range
        for i in range(4):
            for j in range(4):
                assert 0.0 <= distance_matrix[i, j] <= 1.0
    

    def test_calculate_distance_matrix_spoa_failure(self):
        """Test that RuntimeError is raised when SPOA fails."""
        from unittest.mock import patch

        sequences = ["ATCG", "ATCC", "TACG"]

        # Mock run_spoa_msa to return None (failure)
        with patch('gaphack.utils.run_spoa_msa', return_value=None):
            with pytest.raises(RuntimeError) as exc_info:
                calculate_distance_matrix(sequences)

            # Verify error message mentions SPOA failure
            assert "Failed to create multiple sequence alignment" in str(exc_info.value)
            assert "SPOA" in str(exc_info.value)


class TestOutputFormatting:
    """Test output formatting functions."""
    
    def test_format_cluster_output(self):
        """Test formatting of cluster output."""
        clusters = [[0, 1, 2], [3, 4]]
        singletons = [5, 6]
        headers = ["seq_A", "seq_B", "seq_C", "seq_D", "seq_E", "seq_F", "seq_G"]
        
        output = format_cluster_output(clusters, singletons, headers)
        
        assert "Cluster 1 (3 sequences):" in output
        assert "seq_A" in output
        assert "seq_B" in output
        assert "seq_C" in output
        assert "Cluster 2 (2 sequences):" in output
        assert "seq_D" in output
        assert "seq_E" in output
        assert "Singletons (2 sequences):" in output
        assert "seq_F" in output
        assert "seq_G" in output
    
    def test_save_clusters_tsv(self):
        """Test saving clusters in TSV format."""
        clusters = [[0, 1], [2]]
        singletons = [3]
        headers = ["seq1", "seq2", "seq3", "seq4"]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False) as f:
            temp_path = f.name
        
        try:
            save_clusters_to_file(clusters, singletons, temp_path, headers, format="tsv")
            
            # Read and verify the file
            with open(temp_path, 'r') as f:
                lines = f.readlines()
            
            assert lines[0].strip() == "sequence_id\tcluster_id"
            assert "seq1\tcluster_1" in [line.strip() for line in lines]
            assert "seq2\tcluster_1" in [line.strip() for line in lines]
            assert "seq3\tcluster_2" in [line.strip() for line in lines]
            assert "seq4\tsingleton" in [line.strip() for line in lines]
        finally:
            Path(temp_path).unlink()


class TestSequenceValidation:
    """Test sequence validation functions."""
    
    def test_validate_valid_sequences(self):
        """Test validation of valid DNA sequences."""
        sequences = ["ATCG", "GCTA", "AAATTTCCCGGG", "ATCG-N"]
        is_valid, errors = validate_sequences(sequences)
        
        assert is_valid
        assert len(errors) == 0
    
    def test_validate_invalid_sequences(self):
        """Test validation of invalid DNA sequences."""
        sequences = ["ATCG", "GCTX", "AAA@TTT"]
        is_valid, errors = validate_sequences(sequences)
        
        assert not is_valid
        assert len(errors) == 2
        assert "invalid characters" in errors[0].lower()
        assert "invalid characters" in errors[1].lower()
    
    def test_validate_empty_sequences(self):
        """Test validation with empty sequences."""
        sequences = ["ATCG", "", "GCTA"]
        is_valid, errors = validate_sequences(sequences)
        
        assert not is_valid
        assert len(errors) == 1
        assert "empty" in errors[0].lower()
    
    def test_validate_no_sequences(self):
        """Test validation with no sequences."""
        sequences = []
        is_valid, errors = validate_sequences(sequences)

        assert not is_valid
        assert len(errors) == 1
        assert "no sequences" in errors[0].lower()


class TestMSADistanceMinimumOverlap:
    """Test minimum overlap fraction in MSA distance computation."""

    def test_sufficient_overlap_returns_valid_distance(self):
        """Test that sequences with sufficient overlap return valid distance."""
        # Create aligned sequences with good overlap
        # Sequence 1: 100bp original (no terminal gaps)
        # Sequence 2: 100bp original (no terminal gaps)
        # Overlap: 90bp (90% of shorter sequence - well above 50% threshold)
        seq1 = "A" * 90 + "." * 10  # 90bp sequence, 10bp terminal gap at end
        seq2 = "." * 10 + "A" * 90  # 90bp sequence, 10bp terminal gap at start

        distance = compute_msa_distance(seq1, seq2)

        # Should return valid distance (not NaN)
        assert not np.isnan(distance)
        assert 0.0 <= distance <= 1.0

    def test_insufficient_overlap_returns_nan(self):
        """Test that sequences with insufficient overlap return NaN."""
        # Create aligned sequences with minimal overlap
        # Sequence 1: 100bp original
        # Sequence 2: 100bp original
        # Overlap: 40bp (40% of shorter sequence - below 50% threshold)
        seq1 = "A" * 100 + "." * 60
        seq2 = "." * 60 + "A" * 100

        # Only 40bp overlap in the middle
        seq1 = "A" * 40 + "." * 120
        seq2 = "." * 120 + "A" * 40

        distance = compute_msa_distance(seq1, seq2)

        # Should return NaN for insufficient overlap
        assert np.isnan(distance)

    def test_asymmetric_sequences_its2_scenario(self):
        """Test asymmetric sequences like ITS2 vs full ITS."""
        # ITS2: 200bp
        # Full ITS: 500bp
        # Overlap: 200bp (100% of ITS2, 40% of full ITS)
        # Should pass because overlap >= 50% of shorter sequence (200bp)

        # ITS2 fully covered, rest of full ITS is terminal gaps
        its2_aligned = "A" * 200 + "." * 300  # 200bp + 300bp terminal gaps
        full_its_aligned = "A" * 200 + "T" * 300  # Full 500bp sequence

        distance = compute_msa_distance(its2_aligned, full_its_aligned)

        # Should return valid distance (overlap is 200bp, >= 100bp threshold)
        assert not np.isnan(distance)

    def test_asymmetric_sequences_insufficient_overlap(self):
        """Test asymmetric sequences with insufficient overlap."""
        # Short seq: 100bp
        # Long seq: 300bp
        # Overlap: 40bp (40% of short seq - below 50% threshold)

        short_aligned = "A" * 40 + "." * 260
        long_aligned = "." * 60 + "A" * 40 + "T" * 200

        distance = compute_msa_distance(short_aligned, long_aligned)

        # Should return NaN (overlap 40bp < 50bp threshold)
        assert np.isnan(distance)

    def test_exact_threshold_boundary_just_below(self):
        """Test sequences just below the 50% threshold."""
        # Both 100bp original length, overlap 49bp (49% - just below 50% threshold)
        seq1 = "A" * 49 + "T" * 51 + "." * 51  # 100bp + 51 terminal gaps
        seq2 = "." * 51 + "A" * 49 + "C" * 51  # 51 terminal gaps + 100bp

        distance = compute_msa_distance(seq1, seq2)

        # Should return NaN (49bp < 50bp threshold)
        assert np.isnan(distance)

    def test_exact_threshold_boundary_at_threshold(self):
        """Test sequences exactly at the 50% threshold."""
        # Both 100bp original length, overlap 50bp (exactly 50%)
        # Need to create sequences that actually overlap in the middle
        seq1 = "A" * 50 + "T" * 50 + "." * 50  # 100bp + 50 terminal gaps
        seq2 = "." * 50 + "A" * 50 + "C" * 50  # 50 terminal gaps + 100bp

        distance = compute_msa_distance(seq1, seq2)

        # Should return valid distance (overlap is 50bp which is exactly 50% of 100bp)
        assert not np.isnan(distance)

    def test_empty_sequence_returns_nan(self):
        """Test that empty sequences return NaN."""
        # All terminal gaps = empty sequence
        seq1 = "." * 100
        seq2 = "A" * 100

        distance = compute_msa_distance(seq1, seq2)

        assert np.isnan(distance)

    def test_both_empty_sequences_return_nan(self):
        """Test that two empty sequences return NaN."""
        seq1 = "." * 100
        seq2 = "." * 100

        distance = compute_msa_distance(seq1, seq2)

        assert np.isnan(distance)

    def test_no_overlap_after_filtering_returns_nan(self):
        """Test sequences with no overlap after position filtering."""
        # Sequences don't overlap at all
        seq1 = "A" * 100 + "." * 100
        seq2 = "." * 100 + "A" * 100

        distance = compute_msa_distance(seq1, seq2)

        # Should return NaN (zero overlap)
        assert np.isnan(distance)

    def test_internal_gaps_counted_in_original_length(self):
        """Test that internal gaps are counted in original sequence length."""
        # Sequence with internal gaps should count gaps in original length
        # Original length = bases + internal gaps (exclude only terminal gaps)

        # 50 bases + 50 internal gaps + 100 terminal gaps = 100bp original length
        seq1 = "A-" * 25 + "T-" * 25 + "." * 100  # 50 bases, 50 internal gaps
        seq2 = "A-" * 25 + "T-" * 25 + "C" * 100  # 50 bases, 50 internal gaps, then 100 more bases

        distance = compute_msa_distance(seq1, seq2)

        # Original lengths: both have 100bp (50 bases + 50 internal gaps)
        # Overlap after filtering should be considered
        # Should return valid distance if overlap >= 50bp
        assert not np.isnan(distance)

    def test_diverse_sequences_with_sufficient_overlap(self):
        """Test that diverse sequences still get scored if overlap is sufficient."""
        # Create sequences with mutations but sufficient overlap
        # Using real MSA alignment output format

        # Run through SPOA to get realistic aligned sequences
        sequences = [
            "ATCGATCGATCG" * 10,  # 120bp
            "ATCGATCGATCG" * 10,  # 120bp, identical
        ]

        aligned = run_spoa_msa(sequences)
        if aligned is not None:
            aligned = replace_terminal_gaps(aligned)
            distance = compute_msa_distance(aligned[0], aligned[1])

            # Identical sequences should have distance 0
            assert not np.isnan(distance)
            assert distance == 0.0

    def test_original_length_calculation_excludes_gaps(self):
        """Test that original length calculation correctly excludes both gap types."""
        # Create sequences where alignment adds internal gaps
        # Aligned sequences must be same length (as they would from SPOA)

        # When original lengths are NOT provided, should count only bases (not '-' or '.')
        # Both 72 chars total (aligned length)
        seq1_aligned = "ATCG" + "-" * 10 + "ATCG" + "." * 50  # 8 bases, 10 internal gaps, 50 terminal = 68 chars
        seq2_aligned = "." * 10 + "ATCGATCG" + "-" * 5 + "ATCG" + "." * 41  # 12 bases, 5 internal gaps, 51 terminal = 68 chars

        # Calculated original lengths (excluding all gaps):
        # seq1: 8 bases
        # seq2: 12 bases
        # min = 8, threshold = 4bp
        # After filter_msa_positions, overlap should include the overlapping ATCG region
        # As long as overlap >= 4bp, should pass

        distance = compute_msa_distance(seq1_aligned, seq2_aligned)

        # Should return valid distance (overlap sufficient relative to calculated original lengths)
        assert not np.isnan(distance)

    def test_original_length_parameter_overrides_calculation(self):
        """Test that providing original lengths overrides calculation from aligned sequence."""
        # Create aligned sequences
        seq1_aligned = "ATCG" + "." * 96  # Only 4 bases visible
        seq2_aligned = "ATCG" + "." * 96  # Only 4 bases visible

        # Without original lengths: would calculate as 4bp each, threshold = 2bp, overlap = 4bp → PASS
        # With original lengths of 100bp each: threshold = 50bp, overlap = 4bp → FAIL

        # Test with provided original lengths (should fail)
        distance_with_lengths = compute_msa_distance(
            seq1_aligned, seq2_aligned,
            original_len1=100,
            original_len2=100
        )
        assert np.isnan(distance_with_lengths), "Should fail with 100bp original lengths (4bp < 50bp threshold)"

        # Test without provided lengths (should pass)
        distance_without_lengths = compute_msa_distance(seq1_aligned, seq2_aligned)
        assert not np.isnan(distance_without_lengths), "Should pass when calculated as 4bp original (4bp >= 2bp threshold)"
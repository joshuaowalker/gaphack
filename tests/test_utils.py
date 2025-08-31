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
    validate_sequences
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
            sequences, headers = load_sequences_from_fasta(temp_path)
            
            assert len(sequences) == 3
            assert len(headers) == 3
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
    
    def test_alignment_methods(self):
        """Test different alignment methods."""
        sequences = ["ATCG", "ATCC"]
        
        # Test adjusted method
        distance_matrix_adj = calculate_distance_matrix(sequences, alignment_method="adjusted")
        assert distance_matrix_adj.shape == (2, 2)
        assert distance_matrix_adj[0, 0] == 0.0
        assert distance_matrix_adj[1, 1] == 0.0
        
        # Test traditional method
        distance_matrix_trad = calculate_distance_matrix(sequences, alignment_method="traditional")
        assert distance_matrix_trad.shape == (2, 2)
        assert distance_matrix_trad[0, 0] == 0.0
        assert distance_matrix_trad[1, 1] == 0.0
    


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
"""
Tests for BLAST result analysis functionality.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock
import numpy as np

from Bio import SeqIO

from gaphack.blast_analysis import (
    BlastAnalyzer,
    BlastAnalysisResult,
    SequenceResult,
    HistogramData,
    format_text_output,
    format_tsv_output,
    _distance_to_identity,
    _build_histogram
)
from gaphack.utils import MSADistanceResult


# Path to test data
TEST_DATA_DIR = Path(__file__).parent / "test_data"


def create_mock_distance_provider(distance_matrix):
    """Create a mock distance provider that returns distances from a matrix."""
    mock_provider = MagicMock()
    mock_provider.normalization_length = 100
    n = len(distance_matrix)

    def get_distance(i, j):
        return distance_matrix[i][j]

    def get_distance_detailed(i, j):
        dist = distance_matrix[i][j]
        return MSADistanceResult(
            distance_normalized=dist,
            distance_pairwise=dist,
            mismatches=int(dist * 100),
            pairwise_overlap=100,
            is_valid=True
        )

    mock_provider.get_distance.side_effect = get_distance
    mock_provider.get_distance_detailed.side_effect = get_distance_detailed
    return mock_provider


class TestBlastAnalyzer:
    """Test suite for BlastAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = BlastAnalyzer(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=100,
            show_progress=False
        )

    def test_basic_clustering(self):
        """Test happy path with mocked distances: query + 3 conspecific + 6 different.

        Uses mock distance provider to ensure predictable clustering behavior.
        """
        # Create distance matrix: 10 sequences
        # - Sequences 0-3 are close (distances < 0.02)
        # - Sequences 4-9 are far from 0-3 (distances > 0.05)
        distance_matrix = np.zeros((10, 10))

        # Close cluster (0-3): distances around 0.01
        for i in range(4):
            for j in range(4):
                if i != j:
                    distance_matrix[i][j] = 0.01

        # Far cluster (4-9): distances around 0.01 within group
        for i in range(4, 10):
            for j in range(4, 10):
                if i != j:
                    distance_matrix[i][j] = 0.01

        # Distance between clusters: > 0.05
        for i in range(4):
            for j in range(4, 10):
                distance_matrix[i][j] = 0.10
                distance_matrix[j][i] = 0.10

        sequences = [f"SEQ{i}" for i in range(10)]
        headers = [f"seq{i}" for i in range(10)]

        with patch('gaphack.blast_analysis.MSACachedDistanceProvider') as mock_class:
            mock_provider = create_mock_distance_provider(distance_matrix)
            mock_class.return_value = mock_provider

            with patch('gaphack.blast_analysis.TargetModeClustering') as mock_clustering_class:
                mock_clustering = MagicMock()
                mock_clustering_class.return_value = mock_clustering
                # Return cluster containing sequences 0-3
                mock_clustering.cluster.return_value = (
                    [0, 1, 2, 3],  # target cluster
                    [4, 5, 6, 7, 8, 9],  # remaining
                    {'best_config': {'gap_size': 0.05}}
                )

                result = self.analyzer.analyze(sequences, headers)

        assert result.query_id == "seq0"
        assert result.total_sequences == 10
        assert result.query_cluster_size == 4
        assert result.sequences[0].in_query_cluster is True
        assert result.barcode_gap_found is True
        assert result.gap_size_percent is not None

    def test_single_sequence(self):
        """Test edge case: only query, no hits."""
        sequences = ["ATCGATCGATCG"]
        headers = ["query"]

        result = self.analyzer.analyze(sequences, headers)

        assert result.query_id == "query"
        assert result.total_sequences == 1
        assert result.query_cluster_size == 1
        assert result.barcode_gap_found is False
        assert result.gap_size_percent is None
        assert len(result.warnings) > 0
        assert "Insufficient data" in result.warnings[0]
        assert result.sequences[0].in_query_cluster is True
        assert result.sequences[0].identity_to_query == 100.0

    def test_two_sequences_same_cluster(self):
        """Test edge case: query + 1 very similar sequence (via mocked distances)."""
        distance_matrix = np.array([
            [0.0, 0.005],  # Very close - within min_split
            [0.005, 0.0]
        ])

        sequences = ["SEQ0", "SEQ1"]
        headers = ["query", "hit1"]

        with patch('gaphack.blast_analysis.MSACachedDistanceProvider') as mock_class:
            mock_provider = create_mock_distance_provider(distance_matrix)
            mock_class.return_value = mock_provider

            with patch('gaphack.blast_analysis.TargetModeClustering') as mock_clustering_class:
                mock_clustering = MagicMock()
                mock_clustering_class.return_value = mock_clustering
                mock_clustering.cluster.return_value = (
                    [0, 1],  # Both in target cluster
                    [],  # No remaining
                    {'best_config': {'gap_size': 0}}
                )

                result = self.analyzer.analyze(sequences, headers)

        assert result.query_id == "query"
        assert result.total_sequences == 2
        assert result.query_cluster_size == 2
        assert result.barcode_gap_found is False
        assert result.nearest_non_member_identity is None
        for seq in result.sequences:
            assert seq.in_query_cluster is True

    def test_two_sequences_different_clusters(self):
        """Test edge case: query + 1 distant sequence (via mocked distances)."""
        distance_matrix = np.array([
            [0.0, 0.15],  # Very far - beyond max_lump
            [0.15, 0.0]
        ])

        sequences = ["SEQ0", "SEQ1"]
        headers = ["query", "hit1"]

        with patch('gaphack.blast_analysis.MSACachedDistanceProvider') as mock_class:
            mock_provider = create_mock_distance_provider(distance_matrix)
            mock_class.return_value = mock_provider

            with patch('gaphack.blast_analysis.TargetModeClustering') as mock_clustering_class:
                mock_clustering = MagicMock()
                mock_clustering_class.return_value = mock_clustering
                mock_clustering.cluster.return_value = (
                    [0],  # Only query in target cluster
                    [1],  # Other is remaining
                    {'best_config': {'gap_size': 0.10}}
                )

                result = self.analyzer.analyze(sequences, headers)

        assert result.query_id == "query"
        assert result.total_sequences == 2
        assert result.query_cluster_size == 1
        assert result.barcode_gap_found is True
        assert result.sequences[0].in_query_cluster is True
        assert result.sequences[1].in_query_cluster is False

    def test_no_gap_all_same(self):
        """Test all sequences cluster together, no barcode gap (via mocked distances)."""
        # All very close distances
        distance_matrix = np.array([
            [0.0, 0.003, 0.004, 0.003],
            [0.003, 0.0, 0.005, 0.004],
            [0.004, 0.005, 0.0, 0.003],
            [0.003, 0.004, 0.003, 0.0]
        ])

        sequences = ["SEQ0", "SEQ1", "SEQ2", "SEQ3"]
        headers = ["query", "hit1", "hit2", "hit3"]

        with patch('gaphack.blast_analysis.MSACachedDistanceProvider') as mock_class:
            mock_provider = create_mock_distance_provider(distance_matrix)
            mock_class.return_value = mock_provider

            with patch('gaphack.blast_analysis.TargetModeClustering') as mock_clustering_class:
                mock_clustering = MagicMock()
                mock_clustering_class.return_value = mock_clustering
                mock_clustering.cluster.return_value = (
                    [0, 1, 2, 3],  # All in target cluster
                    [],  # No remaining
                    {'best_config': {'gap_size': 0}}
                )

                result = self.analyzer.analyze(sequences, headers)

        assert result.query_cluster_size == 4
        assert result.barcode_gap_found is False
        assert result.nearest_non_member_identity is None
        for seq in result.sequences:
            assert seq.in_query_cluster is True

    def test_alignment_failure(self):
        """Test MSA fails, returns error result with warning."""
        sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA"]
        headers = ["query", "hit1"]

        with patch('gaphack.blast_analysis.MSACachedDistanceProvider') as mock_provider:
            from gaphack.utils import MSAAlignmentError
            mock_provider.side_effect = MSAAlignmentError("SPOA failed")

            result = self.analyzer.analyze(sequences, headers)

            assert result.query_id == "query"
            assert result.query_cluster_size == 1
            assert result.barcode_gap_found is False
            assert "MSA alignment failed" in result.warnings[0]

    def test_empty_input(self):
        """Test no sequences provided."""
        sequences = []
        headers = []

        result = self.analyzer.analyze(sequences, headers)

        assert result.total_sequences == 0
        assert result.query_cluster_size == 0
        assert result.barcode_gap_found is False
        assert len(result.warnings) > 0

    def test_identity_calculations(self):
        """Verify pairwise vs normalized identity math."""
        # Test the _distance_to_identity helper
        assert _distance_to_identity(0.0) == 100.0
        assert _distance_to_identity(0.01) == 99.0
        assert _distance_to_identity(0.1) == 90.0
        assert _distance_to_identity(0.5) == 50.0
        assert _distance_to_identity(1.0) == 0.0
        assert _distance_to_identity(None) is None
        assert _distance_to_identity(float('nan')) is None

    def test_medoid_calculation(self):
        """Verify medoid is most central sequence (via mocked distances)."""
        # Sequence 2 should be medoid (lowest total distance to others)
        distance_matrix = np.array([
            [0.0, 0.02, 0.01, 0.02],  # total = 0.05
            [0.02, 0.0, 0.01, 0.02],  # total = 0.05
            [0.01, 0.01, 0.0, 0.01],  # total = 0.03 <- medoid
            [0.02, 0.02, 0.01, 0.0],  # total = 0.05
        ])

        sequences = ["SEQ0", "SEQ1", "SEQ2", "SEQ3"]
        headers = ["query", "hit1", "hit2", "hit3"]

        with patch('gaphack.blast_analysis.MSACachedDistanceProvider') as mock_class:
            mock_provider = create_mock_distance_provider(distance_matrix)
            mock_class.return_value = mock_provider

            with patch('gaphack.blast_analysis.TargetModeClustering') as mock_clustering_class:
                mock_clustering = MagicMock()
                mock_clustering_class.return_value = mock_clustering
                mock_clustering.cluster.return_value = (
                    [0, 1, 2, 3],
                    [],
                    {'best_config': {'gap_size': 0}}
                )

                result = self.analyzer.analyze(sequences, headers)

        assert result.medoid_id is not None
        assert result.medoid_index == 2  # Sequence 2 has lowest total distance

    def test_nearest_non_member(self):
        """Verify nearest non-member identity calculation (via mocked distances)."""
        # Clear separation: 0,1 in cluster, 2,3 outside
        # Nearest non-member to cluster is sequence 2 at distance 0.05
        distance_matrix = np.array([
            [0.0, 0.01, 0.05, 0.08],
            [0.01, 0.0, 0.06, 0.09],
            [0.05, 0.06, 0.0, 0.01],
            [0.08, 0.09, 0.01, 0.0]
        ])

        sequences = ["SEQ0", "SEQ1", "SEQ2", "SEQ3"]
        headers = ["query", "hit1", "hit2", "hit3"]

        with patch('gaphack.blast_analysis.MSACachedDistanceProvider') as mock_class:
            mock_provider = create_mock_distance_provider(distance_matrix)
            mock_class.return_value = mock_provider

            with patch('gaphack.blast_analysis.TargetModeClustering') as mock_clustering_class:
                mock_clustering = MagicMock()
                mock_clustering_class.return_value = mock_clustering
                mock_clustering.cluster.return_value = (
                    [0, 1],
                    [2, 3],
                    {'best_config': {'gap_size': 0.04}}
                )

                result = self.analyzer.analyze(sequences, headers)

        assert result.query_cluster_size == 2
        assert result.nearest_non_member_identity is not None
        # Distance 0.05 = 95% identity
        assert result.nearest_non_member_identity == 95.0

        # Cluster members should have identity to nearest non-member
        assert result.sequences[0].identity_to_nearest_non_member_normalized is not None
        assert result.sequences[1].identity_to_nearest_non_member_normalized is not None
        # Non-members should not have this field set
        assert result.sequences[2].identity_to_nearest_non_member_normalized is None
        assert result.sequences[3].identity_to_nearest_non_member_normalized is None

    def test_invalid_parameters(self):
        """Test parameter validation."""
        with pytest.raises(ValueError, match="min_split must be non-negative"):
            BlastAnalyzer(min_split=-0.01, max_lump=0.02)

        with pytest.raises(ValueError, match="max_lump must be non-negative"):
            BlastAnalyzer(min_split=0.005, max_lump=-0.02)

        with pytest.raises(ValueError, match="min_split must be less than max_lump"):
            BlastAnalyzer(min_split=0.03, max_lump=0.02)

        with pytest.raises(ValueError, match="min_split must be less than max_lump"):
            BlastAnalyzer(min_split=0.02, max_lump=0.02)


class TestOutputFormats:
    """Test suite for output formatting functions."""

    def setup_method(self):
        """Set up test fixtures with a sample result."""
        self.sample_result = BlastAnalysisResult(
            query_id="query_seq",
            query_length=100,
            total_sequences=5,
            query_cluster_size=3,
            barcode_gap_found=True,
            gap_size_percent=2.5,
            medoid_id="query_seq",
            medoid_index=0,
            intra_cluster_identity={
                "min": 98.5,
                "p5": 98.7,
                "median": 99.0,
                "p95": 99.5,
                "max": 100.0
            },
            nearest_non_member_identity=95.5,
            sequences=[
                SequenceResult(
                    index=0, id="query_seq", in_query_cluster=True,
                    identity_to_query=100.0, identity_to_query_normalized=100.0,
                    identity_to_medoid_normalized=100.0, identity_to_nearest_non_member_normalized=95.5
                ),
                SequenceResult(
                    index=1, id="hit1", in_query_cluster=True,
                    identity_to_query=99.5, identity_to_query_normalized=99.5,
                    identity_to_medoid_normalized=99.5, identity_to_nearest_non_member_normalized=95.2
                ),
                SequenceResult(
                    index=2, id="hit2", in_query_cluster=True,
                    identity_to_query=98.5, identity_to_query_normalized=98.5,
                    identity_to_medoid_normalized=98.5, identity_to_nearest_non_member_normalized=95.0
                ),
                SequenceResult(
                    index=3, id="hit3", in_query_cluster=False,
                    identity_to_query=95.5, identity_to_query_normalized=95.5,
                    identity_to_medoid_normalized=95.0, identity_to_nearest_non_member_normalized=None
                ),
                SequenceResult(
                    index=4, id="hit4", in_query_cluster=False,
                    identity_to_query=90.0, identity_to_query_normalized=90.0,
                    identity_to_medoid_normalized=89.5, identity_to_nearest_non_member_normalized=None
                ),
            ],
            method="gap-optimized-target-clustering",
            min_split=0.005,
            max_lump=0.02,
            normalization_length=100,
            distance_metric="MycoBLAST-adjusted",
            warnings=[]
        )

    def test_json_output_schema(self):
        """Verify JSON structure matches documentation."""
        json_dict = self.sample_result.to_dict()

        # Check top-level structure
        assert "query" in json_dict
        assert "summary" in json_dict
        assert "sequences" in json_dict
        assert "diagnostics" in json_dict

        # Check query section
        assert json_dict["query"]["id"] == "query_seq"
        assert json_dict["query"]["length"] == 100

        # Check summary section
        summary = json_dict["summary"]
        assert summary["total_sequences"] == 5
        assert summary["query_cluster_size"] == 3
        assert summary["barcode_gap_found"] is True
        assert summary["gap_size_percent"] == 2.5
        assert summary["medoid_id"] == "query_seq"
        assert summary["medoid_index"] == 0
        assert "intra_cluster_identity" in summary
        assert "nearest_non_member_identity" in summary

        # Check intra_cluster_identity has all percentiles
        intra = summary["intra_cluster_identity"]
        assert "min" in intra
        assert "p5" in intra
        assert "median" in intra
        assert "p95" in intra
        assert "max" in intra

        # Check sequences section
        assert len(json_dict["sequences"]) == 5
        for seq in json_dict["sequences"]:
            assert "index" in seq
            assert "id" in seq
            assert "in_query_cluster" in seq
            assert "identity_to_query" in seq
            assert "identity_to_query_normalized" in seq
            assert "identity_to_medoid_normalized" in seq
            assert "identity_to_nearest_non_member_normalized" in seq

        # Check diagnostics section
        diag = json_dict["diagnostics"]
        assert diag["method"] == "gap-optimized-target-clustering"
        assert diag["min_split"] == 0.005
        assert diag["max_lump"] == 0.02
        assert "normalization_length" in diag
        assert "identity_metric" in diag
        assert "warnings" in diag

        # Verify it's valid JSON by round-tripping
        json_str = json.dumps(json_dict, indent=2)
        parsed = json.loads(json_str)
        assert parsed == json_dict

    def test_tsv_output_format(self):
        """Verify TSV has correct columns."""
        tsv = format_tsv_output(self.sample_result)
        lines = tsv.strip().split("\n")

        # Check header
        header = lines[0]
        expected_columns = [
            "index", "id", "in_query_cluster",
            "identity_to_query", "identity_to_query_normalized",
            "identity_to_medoid_normalized", "identity_to_nearest_non_member_normalized"
        ]
        assert header == "\t".join(expected_columns)

        # Check data rows
        assert len(lines) == 6  # header + 5 sequences

        # Parse first data row
        row1 = lines[1].split("\t")
        assert row1[0] == "0"  # index
        assert row1[1] == "query_seq"  # id
        assert row1[2] == "true"  # in_query_cluster
        assert row1[3] == "100.00"  # identity_to_query

        # Parse a non-member row
        row4 = lines[4].split("\t")
        assert row4[2] == "false"  # not in_query_cluster

    def test_text_output_format(self):
        """Verify text output is parseable."""
        text = format_text_output(self.sample_result)

        # Check key sections exist
        assert "BLAST Analysis Results" in text
        assert "Query: query_seq" in text
        assert "Query length: 100 bp" in text
        assert "Summary" in text
        assert "Total sequences: 5" in text
        assert "Query cluster size: 3" in text
        assert "Barcode gap found: Yes" in text
        assert "Gap size: 2.50%" in text
        assert "Sequence Classifications" in text
        assert "Diagnostics" in text

        # Check medoid information
        assert "Cluster medoid:" in text
        assert "query_seq" in text

        # Check sequences table has header
        assert "Idx" in text
        assert "ID" in text
        assert "Member" in text

        # Check formatting is consistent (separators)
        assert "=" * 60 in text
        assert "-" * 40 in text


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_sequences_with_none_identities(self):
        """Test handling of sequences with insufficient overlap."""
        result = BlastAnalysisResult(
            query_id="query",
            query_length=100,
            total_sequences=2,
            query_cluster_size=1,
            barcode_gap_found=False,
            gap_size_percent=None,
            medoid_id="query",
            medoid_index=0,
            intra_cluster_identity={"min": None, "p5": None, "median": None, "p95": None, "max": None},
            nearest_non_member_identity=None,
            sequences=[
                SequenceResult(
                    index=0, id="query", in_query_cluster=True,
                    identity_to_query=100.0, identity_to_query_normalized=100.0,
                    identity_to_medoid_normalized=100.0, identity_to_nearest_non_member_normalized=None
                ),
                SequenceResult(
                    index=1, id="hit1", in_query_cluster=False,
                    identity_to_query=None, identity_to_query_normalized=None,
                    identity_to_medoid_normalized=None, identity_to_nearest_non_member_normalized=None
                ),
            ],
            method="gap-optimized-target-clustering",
            min_split=0.005,
            max_lump=0.02,
            normalization_length=100,
            distance_metric="MycoBLAST-adjusted",
            warnings=["1 sequences had insufficient overlap for distance calculation"]
        )

        # JSON output should handle None values
        json_dict = result.to_dict()
        assert json_dict["sequences"][1]["identity_to_query"] is None

        # TSV output should handle None values (empty string)
        tsv = format_tsv_output(result)
        lines = tsv.split("\n")  # Don't strip - preserve trailing tabs
        assert len(lines) == 3  # header + 2 sequences
        # Check that the second sequence row contains empty values for identity
        assert "\tfalse\t\t\t\t" in lines[2]  # Empty identity columns

        # Text output should handle None values
        text = format_text_output(result)
        assert "N/A" in text

    def test_result_with_warnings(self):
        """Test that warnings are properly included in output."""
        result = BlastAnalysisResult(
            query_id="query",
            query_length=100,
            total_sequences=1,
            query_cluster_size=1,
            barcode_gap_found=False,
            gap_size_percent=None,
            medoid_id="query",
            medoid_index=0,
            intra_cluster_identity={"min": None, "p5": None, "median": None, "p95": None, "max": None},
            nearest_non_member_identity=None,
            sequences=[
                SequenceResult(
                    index=0, id="query", in_query_cluster=True,
                    identity_to_query=100.0, identity_to_query_normalized=100.0,
                    identity_to_medoid_normalized=100.0, identity_to_nearest_non_member_normalized=None
                ),
            ],
            method="gap-optimized-target-clustering",
            min_split=0.005,
            max_lump=0.02,
            normalization_length=100,
            distance_metric="MycoBLAST-adjusted",
            warnings=["Query has no close matches", "Test warning"]
        )

        # Check JSON output
        json_dict = result.to_dict()
        assert len(json_dict["diagnostics"]["warnings"]) == 2
        assert "Query has no close matches" in json_dict["diagnostics"]["warnings"]

        # Check text output
        text = format_text_output(result)
        assert "Warnings:" in text
        assert "Query has no close matches" in text
        assert "Test warning" in text

    def test_rounding_in_json_output(self):
        """Test that JSON output properly rounds identity values."""
        result = BlastAnalysisResult(
            query_id="query",
            query_length=100,
            total_sequences=2,
            query_cluster_size=2,
            barcode_gap_found=True,
            gap_size_percent=1.23456789,
            medoid_id="query",
            medoid_index=0,
            intra_cluster_identity={
                "min": 98.123456,
                "p5": 98.234567,
                "median": 99.345678,
                "p95": 99.456789,
                "max": 99.567890
            },
            nearest_non_member_identity=95.123456,
            sequences=[
                SequenceResult(
                    index=0, id="query", in_query_cluster=True,
                    identity_to_query=100.0, identity_to_query_normalized=100.0,
                    identity_to_medoid_normalized=100.0, identity_to_nearest_non_member_normalized=95.123456
                ),
                SequenceResult(
                    index=1, id="hit1", in_query_cluster=True,
                    identity_to_query=98.765432, identity_to_query_normalized=98.654321,
                    identity_to_medoid_normalized=98.543210, identity_to_nearest_non_member_normalized=95.432109
                ),
            ],
            method="gap-optimized-target-clustering",
            min_split=0.005,
            max_lump=0.02,
            normalization_length=100,
            distance_metric="MycoBLAST-adjusted",
            warnings=[]
        )

        json_dict = result.to_dict()

        # Gap size should be rounded to 2 decimal places
        assert json_dict["summary"]["gap_size_percent"] == 1.23

        # Intra-cluster identities should be rounded
        assert json_dict["summary"]["intra_cluster_identity"]["min"] == 98.12

        # Sequence identities should be rounded
        assert json_dict["sequences"][1]["identity_to_query"] == 98.77


@pytest.mark.integration
class TestBlastAnalyzerIntegration:
    """Integration tests using real biological sequences."""

    def test_blast_analyzer_with_real_sequences(self):
        """Test BlastAnalyzer end-to-end with real Russula ITS sequences.

        Uses russula_diverse_50.fasta which contains sequences from multiple
        Russula species. The first sequence should cluster with other sequences
        of the same species.
        """
        fasta_path = TEST_DATA_DIR / "russula_diverse_50.fasta"
        if not fasta_path.exists():
            pytest.skip(f"Test data not found: {fasta_path}")

        # Load sequences
        sequences = []
        headers = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq).upper())
            headers.append(record.id)

        # Take first 10 sequences for a faster test
        sequences = sequences[:10]
        headers = headers[:10]

        # Run analysis with real implementation (no mocks)
        analyzer = BlastAnalyzer(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=100,
            show_progress=False
        )

        result = analyzer.analyze(sequences, headers)

        # Basic sanity checks
        assert result.query_id == headers[0]
        assert result.total_sequences == 10
        assert result.query_cluster_size >= 1
        assert result.query_cluster_size <= 10

        # Query should always be in its own cluster
        assert result.sequences[0].in_query_cluster is True
        assert result.sequences[0].identity_to_query == 100.0

        # Medoid should be set
        assert result.medoid_id is not None
        assert result.medoid_index is not None
        assert 0 <= result.medoid_index < 10

        # All identity values should be reasonable (0-100% or None)
        for seq in result.sequences:
            if seq.identity_to_query is not None:
                assert 0 <= seq.identity_to_query <= 100
            if seq.identity_to_query_normalized is not None:
                assert 0 <= seq.identity_to_query_normalized <= 100

        # Output formats should work
        json_output = result.to_dict()
        assert "query" in json_output
        assert "summary" in json_output
        assert "sequences" in json_output

        tsv_output = format_tsv_output(result)
        assert len(tsv_output.split("\n")) == 11  # header + 10 sequences

        text_output = format_text_output(result)
        assert "BLAST Analysis Results" in text_output

    def test_blast_analyzer_finds_conspecifics(self):
        """Test that BlastAnalyzer correctly clusters conspecific sequences.

        The first ~15 sequences in russula_diverse_50.fasta are all
        'Russula sp. IN4', so they should cluster together.
        """
        fasta_path = TEST_DATA_DIR / "russula_diverse_50.fasta"
        if not fasta_path.exists():
            pytest.skip(f"Test data not found: {fasta_path}")

        # Load all sequences
        sequences = []
        headers = []
        species = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq).upper())
            headers.append(record.id)
            # Extract species from header (format: name="Russula sp. 'IN4'")
            desc = record.description
            if 'name="' in desc:
                sp = desc.split('name="')[1].split('"')[0]
                species.append(sp)
            else:
                species.append("unknown")

        # Run analysis
        analyzer = BlastAnalyzer(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=100,
            show_progress=False
        )

        result = analyzer.analyze(sequences, headers)

        # The query species
        query_species = species[0]

        # Count how many of the same species are in the query cluster
        same_species_in_cluster = 0
        same_species_total = 0
        for i, seq_result in enumerate(result.sequences):
            if species[i] == query_species:
                same_species_total += 1
                if seq_result.in_query_cluster:
                    same_species_in_cluster += 1

        # Most conspecifics should be in the cluster (allow some flexibility
        # due to sequence quality variation)
        if same_species_total > 1:
            conspecific_ratio = same_species_in_cluster / same_species_total
            assert conspecific_ratio >= 0.5, (
                f"Only {same_species_in_cluster}/{same_species_total} "
                f"conspecifics in cluster (expected >50%)"
            )


class TestHistograms:
    """Test suite for histogram functionality."""

    def test_build_histogram_basic(self):
        """Test basic histogram building."""
        values = [99.0, 99.2, 99.5, 99.7, 100.0]
        histogram = _build_histogram(values, bin_width=0.5)

        assert histogram is not None
        assert histogram.bin_width_percent == 0.5
        # Values span 99.0-100.0, should have bins at 99.0, 99.5, 100.0
        assert len(histogram.bin_starts) > 0
        assert sum(histogram.counts) == 5  # All values accounted for
        # Frequencies should sum to 1.0
        assert abs(sum(histogram.frequencies) - 1.0) < 0.0001

    def test_build_histogram_empty_input(self):
        """Test histogram with empty input."""
        histogram = _build_histogram([])
        assert histogram is None

        histogram = _build_histogram([None, None])
        assert histogram is None

    def test_build_histogram_single_value(self):
        """Test histogram with single value."""
        histogram = _build_histogram([99.5])
        assert histogram is not None
        assert len(histogram.bin_starts) == 1
        assert histogram.counts == [1]
        assert histogram.frequencies == [1.0]

    def test_build_histogram_omits_empty_bins(self):
        """Test that empty bins are omitted."""
        # Values at 95% and 99% with gap in between
        values = [95.0, 95.1, 99.0, 99.1]
        histogram = _build_histogram(values, bin_width=0.5)

        assert histogram is not None
        # Should only have bins at 95.0 and 99.0, not the empty ones in between
        assert 96.0 not in histogram.bin_starts
        assert 97.0 not in histogram.bin_starts
        assert sum(histogram.counts) == 4

    def test_histogram_in_json_output(self):
        """Test that histograms appear in JSON output."""
        result = BlastAnalysisResult(
            query_id="query",
            query_length=100,
            total_sequences=5,
            query_cluster_size=3,
            barcode_gap_found=True,
            gap_size_percent=2.0,
            medoid_id="query",
            medoid_index=0,
            intra_cluster_identity={"min": 98.0, "p5": 98.5, "median": 99.0, "p95": 99.5, "max": 100.0},
            nearest_non_member_identity=96.0,
            sequences=[],
            method="gap-optimized-target-clustering",
            min_split=0.005,
            max_lump=0.02,
            normalization_length=100,
            distance_metric="test",
            warnings=[],
            intra_cluster_histogram=HistogramData(
                bin_width_percent=0.5,
                bin_starts=[99.0, 99.5, 100.0],
                counts=[2, 3, 1],
                frequencies=[2/6, 3/6, 1/6]
            ),
            inter_cluster_histogram=HistogramData(
                bin_width_percent=0.5,
                bin_starts=[95.5, 96.0],
                counts=[1, 2],
                frequencies=[1/3, 2/3]
            )
        )

        json_dict = result.to_dict()

        # Check histogram structure in diagnostics
        assert "histograms" in json_dict["diagnostics"]
        histograms = json_dict["diagnostics"]["histograms"]

        assert "intra_cluster" in histograms
        assert "inter_cluster" in histograms

        intra = histograms["intra_cluster"]
        assert intra["bin_width_percent"] == 0.5
        assert intra["bin_starts"] == [99.0, 99.5, 100.0]
        assert intra["counts"] == [2, 3, 1]
        assert "frequencies" in intra
        assert abs(sum(intra["frequencies"]) - 1.0) < 0.0001

        inter = histograms["inter_cluster"]
        assert inter["bin_starts"] == [95.5, 96.0]
        assert inter["counts"] == [1, 2]
        assert "frequencies" in inter
        assert abs(sum(inter["frequencies"]) - 1.0) < 0.0001

    def test_histogram_none_when_insufficient_data(self):
        """Test that histograms are None for edge cases."""
        result = BlastAnalysisResult(
            query_id="query",
            query_length=100,
            total_sequences=1,
            query_cluster_size=1,
            barcode_gap_found=False,
            gap_size_percent=None,
            medoid_id="query",
            medoid_index=0,
            intra_cluster_identity={"min": None, "p5": None, "median": None, "p95": None, "max": None},
            nearest_non_member_identity=None,
            sequences=[],
            method="gap-optimized-target-clustering",
            min_split=0.005,
            max_lump=0.02,
            normalization_length=100,
            distance_metric="test",
            warnings=[],
            intra_cluster_histogram=None,
            inter_cluster_histogram=None
        )

        json_dict = result.to_dict()
        assert json_dict["diagnostics"]["histograms"]["intra_cluster"] is None
        assert json_dict["diagnostics"]["histograms"]["inter_cluster"] is None

    @pytest.mark.integration
    def test_histogram_integration_with_real_sequences(self):
        """Test histograms are computed correctly with real sequences."""
        fasta_path = TEST_DATA_DIR / "russula_diverse_50.fasta"
        if not fasta_path.exists():
            pytest.skip(f"Test data not found: {fasta_path}")

        # Load first 20 sequences
        sequences = []
        headers = []
        for i, record in enumerate(SeqIO.parse(fasta_path, "fasta")):
            if i >= 20:
                break
            sequences.append(str(record.seq).upper())
            headers.append(record.id)

        analyzer = BlastAnalyzer(
            min_split=0.005,
            max_lump=0.02,
            show_progress=False
        )

        result = analyzer.analyze(sequences, headers)

        # Should have histograms if cluster has multiple members
        if result.query_cluster_size > 1:
            assert result.intra_cluster_histogram is not None
            assert result.intra_cluster_histogram.bin_width_percent == 0.5
            assert len(result.intra_cluster_histogram.bin_starts) > 0
            assert sum(result.intra_cluster_histogram.counts) > 0

            # Number of intra-cluster pairs = n*(n-1)/2
            n = result.query_cluster_size
            expected_pairs = n * (n - 1) // 2
            assert sum(result.intra_cluster_histogram.counts) == expected_pairs

        # Should have inter-cluster histogram if there are non-members
        if result.query_cluster_size < result.total_sequences:
            assert result.inter_cluster_histogram is not None
            # One entry per cluster member
            assert sum(result.inter_cluster_histogram.counts) == result.query_cluster_size

"""Tests for gaphack-refine CLI functionality."""

import pytest
import tempfile
import shutil
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from gaphack.refine_cli import (
    load_clusters_from_directory,
    detect_conflicts,
    generate_cluster_mapping_report,
    write_output_clusters,
    calculate_cluster_statistics
)


@pytest.fixture
def temp_cluster_dir():
    """Create temporary directory for test clusters."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def simple_clusters(temp_cluster_dir):
    """Create simple test clusters without conflicts."""
    # Cluster 1: 3 sequences
    cluster1 = temp_cluster_dir / "cluster_001.fasta"
    records = [
        SeqRecord(Seq("ACGTACGTACGT"), id="seq1", description=""),
        SeqRecord(Seq("ACGTACGTACGG"), id="seq2", description=""),
        SeqRecord(Seq("ACGTACGTACGA"), id="seq3", description=""),
    ]
    with open(cluster1, 'w') as f:
        SeqIO.write(records, f, "fasta")

    # Cluster 2: 2 sequences
    cluster2 = temp_cluster_dir / "cluster_002.fasta"
    records = [
        SeqRecord(Seq("TTTTTTTTTTTT"), id="seq4", description=""),
        SeqRecord(Seq("TTTTTTTTTTGG"), id="seq5", description=""),
    ]
    with open(cluster2, 'w') as f:
        SeqIO.write(records, f, "fasta")

    return temp_cluster_dir


@pytest.fixture
def conflicting_clusters(temp_cluster_dir):
    """Create test clusters with conflicts (shared sequences)."""
    # Cluster A: seq1, seq2, seq3
    cluster_a = temp_cluster_dir / "cluster_A.fasta"
    records = [
        SeqRecord(Seq("ACGTACGTACGT"), id="seq1", description=""),
        SeqRecord(Seq("ACGTACGTACGG"), id="seq2", description=""),
        SeqRecord(Seq("ACGTACGTACGA"), id="seq3", description=""),  # Will be duplicated in cluster_B
    ]
    with open(cluster_a, 'w') as f:
        SeqIO.write(records, f, "fasta")

    # Cluster B: seq3, seq4, seq5 (seq3 is conflict)
    cluster_b = temp_cluster_dir / "cluster_B.fasta"
    records = [
        SeqRecord(Seq("ACGTACGTACGA"), id="seq3", description=""),  # Duplicate!
        SeqRecord(Seq("ACGTACGTACGC"), id="seq4", description=""),
        SeqRecord(Seq("ACGTACGTACGX"), id="seq5", description=""),
    ]
    with open(cluster_b, 'w') as f:
        SeqIO.write(records, f, "fasta")

    # Cluster C: seq6, seq7 (no conflicts)
    cluster_c = temp_cluster_dir / "cluster_C.fasta"
    records = [
        SeqRecord(Seq("TTTTTTTTTTTT"), id="seq6", description=""),
        SeqRecord(Seq("TTTTTTTTTTGG"), id="seq7", description=""),
    ]
    with open(cluster_c, 'w') as f:
        SeqIO.write(records, f, "fasta")

    return temp_cluster_dir


@pytest.fixture
def clusters_with_unassigned(temp_cluster_dir):
    """Create test clusters with unassigned.fasta."""
    # Cluster 1
    cluster1 = temp_cluster_dir / "cluster_001.fasta"
    records = [
        SeqRecord(Seq("ACGTACGTACGT"), id="seq1", description=""),
        SeqRecord(Seq("ACGTACGTACGG"), id="seq2", description=""),
    ]
    with open(cluster1, 'w') as f:
        SeqIO.write(records, f, "fasta")

    # Unassigned
    unassigned = temp_cluster_dir / "unassigned.fasta"
    records = [
        SeqRecord(Seq("NNNNNNNNNNNN"), id="unassigned1", description=""),
        SeqRecord(Seq("NNNNNNNNNNNG"), id="unassigned2", description=""),
    ]
    with open(unassigned, 'w') as f:
        SeqIO.write(records, f, "fasta")

    return temp_cluster_dir


class TestLoadClustersFromDirectory:
    """Tests for load_clusters_from_directory function."""

    def test_load_simple_clusters(self, simple_clusters):
        """Test loading simple clusters without conflicts."""
        clusters, sequences, headers, unassigned, header_mapping = load_clusters_from_directory(simple_clusters)

        assert len(clusters) == 2
        assert "cluster_001" in clusters
        assert "cluster_002" in clusters
        assert len(clusters["cluster_001"]) == 3
        assert len(clusters["cluster_002"]) == 2
        assert len(sequences) == 5
        assert len(headers) == 5
        assert len(unassigned) == 0

    def test_load_conflicting_clusters(self, conflicting_clusters):
        """Test loading clusters with conflicts (duplicated sequences)."""
        clusters, sequences, headers, unassigned, header_mapping = load_clusters_from_directory(conflicting_clusters)

        assert len(clusters) == 3
        assert "cluster_A" in clusters
        assert "cluster_B" in clusters
        assert "cluster_C" in clusters

        # Total headers includes duplicates
        assert len(headers) == 8  # 3 + 3 + 2 = 8 (seq3 counted twice)

        # Check that seq3 appears in both clusters
        assert "seq3" in clusters["cluster_A"]
        assert "seq3" in clusters["cluster_B"]

    def test_load_clusters_with_unassigned(self, clusters_with_unassigned):
        """Test loading clusters with unassigned.fasta."""
        clusters, sequences, headers, unassigned, header_mapping = load_clusters_from_directory(clusters_with_unassigned)

        assert len(clusters) == 1
        assert "cluster_001" in clusters
        assert len(unassigned) == 2
        assert "unassigned1" in unassigned
        assert "unassigned2" in unassigned

        # Unassigned sequences should not be in main sequences/headers
        assert len(headers) == 2  # Only cluster_001 sequences

    def test_load_missing_directory(self):
        """Test error handling for missing directory."""
        with pytest.raises(FileNotFoundError):
            load_clusters_from_directory(Path("/nonexistent/directory"))

    def test_load_empty_directory(self, temp_cluster_dir):
        """Test error handling for directory with no FASTA files."""
        with pytest.raises(ValueError, match="No cluster files"):
            load_clusters_from_directory(temp_cluster_dir)

    def test_skip_empty_cluster_files(self, temp_cluster_dir):
        """Test that empty cluster files are skipped with warning."""
        # Create one valid cluster
        cluster1 = temp_cluster_dir / "cluster_001.fasta"
        records = [SeqRecord(Seq("ACGT"), id="seq1", description="")]
        with open(cluster1, 'w') as f:
            SeqIO.write(records, f, "fasta")

        # Create one empty cluster file
        empty = temp_cluster_dir / "cluster_empty.fasta"
        empty.touch()

        clusters, sequences, headers, unassigned, header_mapping = load_clusters_from_directory(temp_cluster_dir)

        # Should only load the non-empty cluster
        assert len(clusters) == 1
        assert "cluster_001" in clusters
        assert "cluster_empty" not in clusters


class TestDetectConflicts:
    """Tests for detect_conflicts function."""

    def test_no_conflicts(self):
        """Test conflict detection when no conflicts exist."""
        clusters = {
            "cluster_1": ["seq1", "seq2", "seq3"],
            "cluster_2": ["seq4", "seq5"],
        }

        conflicts = detect_conflicts(clusters)

        assert len(conflicts) == 0

    def test_single_conflict(self):
        """Test detecting a single conflicted sequence."""
        clusters = {
            "cluster_A": ["seq1", "seq2", "seq3"],
            "cluster_B": ["seq3", "seq4", "seq5"],  # seq3 is conflict
        }

        conflicts = detect_conflicts(clusters)

        assert len(conflicts) == 1
        assert "seq3" in conflicts
        assert set(conflicts["seq3"]) == {"cluster_A", "cluster_B"}

    def test_multiple_conflicts(self):
        """Test detecting multiple conflicted sequences."""
        clusters = {
            "cluster_A": ["seq1", "seq2", "seq3"],
            "cluster_B": ["seq3", "seq4", "seq5"],  # seq3 conflict
            "cluster_C": ["seq5", "seq6"],  # seq5 conflict with cluster_B
        }

        conflicts = detect_conflicts(clusters)

        assert len(conflicts) == 2
        assert "seq3" in conflicts
        assert "seq5" in conflicts
        assert set(conflicts["seq3"]) == {"cluster_A", "cluster_B"}
        assert set(conflicts["seq5"]) == {"cluster_B", "cluster_C"}

    def test_multi_cluster_conflict(self):
        """Test sequence appearing in 3+ clusters."""
        clusters = {
            "cluster_A": ["seq1", "seq2"],
            "cluster_B": ["seq2", "seq3"],
            "cluster_C": ["seq2", "seq4"],  # seq2 in all three
        }

        conflicts = detect_conflicts(clusters)

        assert len(conflicts) == 1
        assert "seq2" in conflicts
        assert set(conflicts["seq2"]) == {"cluster_A", "cluster_B", "cluster_C"}


class TestWriteOutputClusters:
    """Tests for write_output_clusters function."""

    def test_write_clusters_with_renumbering(self, temp_cluster_dir):
        """Test writing clusters with renumbering by size."""
        clusters = {
            "cluster_A": ["seq1", "seq2", "seq3", "seq4", "seq5"],  # Size 5 (largest)
            "cluster_B": ["seq6"],  # Size 1 (smallest)
            "cluster_C": ["seq7", "seq8", "seq9"],  # Size 3 (middle)
        }

        sequences = [
            "ACGT", "ACGG", "ACGA", "ACGC", "ACGN",  # seq1-5
            "TTTT",  # seq6
            "GGGG", "GGGA", "GGGC",  # seq7-9
        ]
        headers = ["seq1", "seq2", "seq3", "seq4", "seq5", "seq6", "seq7", "seq8", "seq9"]
        header_mapping = {h: h for h in headers}  # Simple mapping for test

        output_dir = temp_cluster_dir / "output"
        output_dir.mkdir()

        id_mapping = write_output_clusters(
            clusters=clusters,
            sequences=sequences,
            headers=headers,
            unassigned_headers=[],
            output_dir=output_dir,
            header_mapping=header_mapping,
            renumber=True
        )

        # Check renumbering (by size, largest first)
        assert id_mapping["cluster_A"] == "cluster_00001"  # Largest
        assert id_mapping["cluster_C"] == "cluster_00002"  # Middle
        assert id_mapping["cluster_B"] == "cluster_00003"  # Smallest

        # Check files exist
        assert (output_dir / "cluster_00001.fasta").exists()
        assert (output_dir / "cluster_00002.fasta").exists()
        assert (output_dir / "cluster_00003.fasta").exists()

        # Check cluster contents
        records = list(SeqIO.parse(output_dir / "cluster_00001.fasta", "fasta"))
        assert len(records) == 5

    def test_write_clusters_preserve_ids(self, temp_cluster_dir):
        """Test writing clusters with preserved IDs (no renumbering)."""
        clusters = {
            "my_cluster_A": ["seq1", "seq2"],
            "vsearch_otu_5": ["seq3"],
        }

        sequences = ["ACGT", "ACGG", "TTTT"]
        headers = ["seq1", "seq2", "seq3"]
        header_mapping = {h: h for h in headers}  # Simple mapping for test

        output_dir = temp_cluster_dir / "output"
        output_dir.mkdir()

        id_mapping = write_output_clusters(
            clusters=clusters,
            sequences=sequences,
            headers=headers,
            unassigned_headers=[],
            output_dir=output_dir,
            header_mapping=header_mapping,
            renumber=False
        )

        # IDs should be preserved
        assert id_mapping["my_cluster_A"] == "my_cluster_A"
        assert id_mapping["vsearch_otu_5"] == "vsearch_otu_5"

        # Check files exist with original names
        assert (output_dir / "my_cluster_A.fasta").exists()
        assert (output_dir / "vsearch_otu_5.fasta").exists()

    def test_write_unassigned_file(self, temp_cluster_dir):
        """Test writing unassigned.fasta."""
        clusters = {"cluster_001": ["seq1"]}
        sequences = ["ACGT", "NNNN"]
        headers = ["seq1", "unassigned1"]
        unassigned_headers = ["unassigned1"]
        header_mapping = {h: h for h in headers}  # Simple mapping for test

        output_dir = temp_cluster_dir / "output"
        output_dir.mkdir()

        write_output_clusters(
            clusters=clusters,
            sequences=sequences,
            headers=headers,
            unassigned_headers=unassigned_headers,
            output_dir=output_dir,
            header_mapping=header_mapping,
            renumber=True
        )

        # Check unassigned file exists
        unassigned_file = output_dir / "unassigned.fasta"
        assert unassigned_file.exists()

        records = list(SeqIO.parse(unassigned_file, "fasta"))
        assert len(records) == 1
        assert records[0].id == "unassigned1"


class TestCalculateClusterStatistics:
    """Tests for calculate_cluster_statistics function."""

    def test_calculate_statistics_typical_distribution(self):
        """Test statistics calculation with typical cluster distribution."""
        clusters = {
            "c1": ["s1", "s2", "s3"],  # Size 3 (1-10)
            "c2": ["s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15"],  # Size 12 (11-50)
            "c3": ["s" + str(i) for i in range(60)],  # Size 60 (51-100)
            "c4": ["s" + str(i) for i in range(150)],  # Size 150 (100+)
        }

        stats = calculate_cluster_statistics(clusters)

        assert stats['1-10'] == 1
        assert stats['11-50'] == 1
        assert stats['51-100'] == 1
        assert stats['100+'] == 1

        # Check percentages sum to 100%
        total_pct = stats['1-10_pct'] + stats['11-50_pct'] + stats['51-100_pct'] + stats['100+_pct']
        assert abs(total_pct - 100.0) < 0.01

    def test_calculate_statistics_all_small_clusters(self):
        """Test statistics when all clusters are small (1-10)."""
        clusters = {
            "c1": ["s1"],
            "c2": ["s2", "s3"],
            "c3": ["s4", "s5", "s6"],
        }

        stats = calculate_cluster_statistics(clusters)

        assert stats['1-10'] == 3
        assert stats['11-50'] == 0
        assert stats['51-100'] == 0
        assert stats['100+'] == 0
        assert stats['1-10_pct'] == 100.0

    def test_calculate_statistics_empty_clusters(self):
        """Test statistics with no clusters."""
        clusters = {}
        stats = calculate_cluster_statistics(clusters)

        assert stats['1-10'] == 0
        assert stats['11-50'] == 0
        assert stats['51-100'] == 0
        assert stats['100+'] == 0


class TestGenerateClusterMappingReport:
    """Tests for generate_cluster_mapping_report function."""

    def test_generate_mapping_simple(self, temp_cluster_dir):
        """Test generating simple cluster mapping report."""
        original_clusters = {
            "cluster_A": ["seq1", "seq2", "seq3"],
            "cluster_B": ["seq4", "seq5"],
        }

        final_clusters = {
            "cluster_00001": ["seq1", "seq2", "seq3"],
            "cluster_00002": ["seq4", "seq5"],
        }

        output_path = temp_cluster_dir / "mapping.txt"

        generate_cluster_mapping_report(
            original_clusters=original_clusters,
            stage1_clusters=None,
            stage2_clusters=None,
            final_clusters=final_clusters,
            output_path=output_path
        )

        assert output_path.exists()

        # Read and check contents
        with open(output_path) as f:
            content = f.read()

        assert "cluster_A → cluster_00001" in content
        assert "cluster_B → cluster_00002" in content
        assert "Total original clusters: 2" in content
        assert "Total final clusters: 2" in content

    def test_generate_mapping_with_merges(self, temp_cluster_dir):
        """Test mapping report when clusters are merged."""
        original_clusters = {
            "cluster_A": ["seq1", "seq2", "seq3"],
            "cluster_B": ["seq3", "seq4", "seq5"],  # Will merge with A (conflict on seq3)
            "cluster_C": ["seq6", "seq7"],
        }

        # After refinement: A and B merged
        final_clusters = {
            "cluster_00001": ["seq1", "seq2", "seq3", "seq4", "seq5"],  # Merged A+B
            "cluster_00002": ["seq6", "seq7"],  # C unchanged
        }

        output_path = temp_cluster_dir / "mapping.txt"

        generate_cluster_mapping_report(
            original_clusters=original_clusters,
            stage1_clusters=None,
            stage2_clusters=None,
            final_clusters=final_clusters,
            output_path=output_path
        )

        assert output_path.exists()

        with open(output_path) as f:
            content = f.read()

        # Both A and B should map to cluster_00001
        assert "cluster_A → cluster_00001" in content
        assert "cluster_B → cluster_00001" in content
        assert "cluster_C → cluster_00002" in content


@pytest.mark.integration
class TestRefineCliIntegration:
    """Integration tests for refine CLI."""

    def test_refine_no_conflicts(self, simple_clusters, temp_cluster_dir):
        """Test refinement when no conflicts exist."""
        output_dir = temp_cluster_dir / "output"

        # Since we're testing the components directly
        clusters, sequences, headers, unassigned, header_mapping = load_clusters_from_directory(simple_clusters)
        conflicts = detect_conflicts(clusters)

        assert len(conflicts) == 0
        assert len(clusters) == 2

    def test_refine_with_conflicts(self, conflicting_clusters, temp_cluster_dir):
        """Test refinement with conflicts present."""
        clusters, sequences, headers, unassigned, header_mapping = load_clusters_from_directory(conflicting_clusters)
        conflicts = detect_conflicts(clusters)

        # Should detect seq3 conflict
        assert len(conflicts) == 1
        assert "seq3" in conflicts
        assert len(conflicts["seq3"]) == 2

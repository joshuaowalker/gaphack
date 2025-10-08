"""Integration tests for gaphack-refine with real workflows."""

import pytest
import tempfile
import shutil
import subprocess
from pathlib import Path
from Bio import SeqIO


@pytest.fixture
def temp_work_dir():
    """Create temporary working directory for integration tests."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def russula_50_fasta():
    """Path to Russula 50 sequence test dataset."""
    return Path("tests/test_data/russula_diverse_50.fasta")


@pytest.mark.integration
class TestDecomposeRefineWorkflow:
    """Test complete decompose → refine workflow."""

    def test_refine_decompose_output_russula_50(self, russula_50_fasta, temp_work_dir):
        """Test refining gaphack-decompose output with Russula 50 dataset."""
        if not russula_50_fasta.exists():
            pytest.skip("Russula 50 dataset not found")

        # Step 1: Run decompose to create initial clusters
        decompose_output = temp_work_dir / "decompose_out"
        result = subprocess.run([
            "gaphack-decompose",
            str(russula_50_fasta),
            "-o", str(decompose_output),
            "--max-clusters", "10",
            "--min-split", "0.005",
            "--max-lump", "0.02"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Decompose failed: {result.stderr}"
        assert decompose_output.exists()

        # Check that clusters were created
        clusters_dir = decompose_output / "clusters" / "latest"
        assert clusters_dir.exists()

        cluster_files = list(clusters_dir.glob("cluster_*.fasta"))
        assert len(cluster_files) > 0, "No cluster files created by decompose"

        print(f"Decompose created {len(cluster_files)} clusters")

        # Step 2: Run refine on decompose output (conflict resolution only)
        refine_output = temp_work_dir / "refined_out"
        result = subprocess.run([
            "gaphack-refine",
            "--input-dir", str(clusters_dir),
            "--output-dir", str(refine_output),
            "--min-split", "0.005",
            "--max-lump", "0.02"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Refine failed: {result.stderr}"
        assert refine_output.exists()

        # Check refined output
        refined_clusters_dir = refine_output / "latest"
        assert refined_clusters_dir.exists()

        refined_cluster_files = list(refined_clusters_dir.glob("cluster_*.fasta"))
        assert len(refined_cluster_files) > 0, "No refined clusters created"

        print(f"Refine created {len(refined_cluster_files)} clusters")

        # Check that summary and mapping files exist
        assert (refined_clusters_dir / "refine_summary.txt").exists()
        assert (refined_clusters_dir / "cluster_mapping.txt").exists()

        # Verify MECE property: all sequences assigned exactly once
        sequence_assignments = {}
        for cluster_file in refined_cluster_files:
            cluster_id = cluster_file.stem
            for record in SeqIO.parse(cluster_file, "fasta"):
                if record.id in sequence_assignments:
                    pytest.fail(f"Sequence {record.id} assigned to multiple clusters: "
                              f"{sequence_assignments[record.id]} and {cluster_id}")
                sequence_assignments[record.id] = cluster_id

        print(f"MECE property verified: {len(sequence_assignments)} sequences, each assigned exactly once")

    def test_refine_with_close_cluster_refinement(self, russula_50_fasta, temp_work_dir):
        """Test refining with close cluster refinement enabled."""
        if not russula_50_fasta.exists():
            pytest.skip("Russula 50 dataset not found")

        # Step 1: Run decompose
        decompose_output = temp_work_dir / "decompose_out"
        result = subprocess.run([
            "gaphack-decompose",
            str(russula_50_fasta),
            "-o", str(decompose_output),
            "--max-clusters", "15",  # Create more clusters for refinement
            "--min-split", "0.005",
            "--max-lump", "0.02"
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Decompose failed: {result.stderr}"

        clusters_dir = decompose_output / "clusters" / "latest"
        initial_clusters = len(list(clusters_dir.glob("cluster_*.fasta")))
        print(f"Initial clusters: {initial_clusters}")

        # Step 2: Run refine with close cluster refinement
        refine_output = temp_work_dir / "refined_out"
        result = subprocess.run([
            "gaphack-refine",
            "--input-dir", str(clusters_dir),
            "--output-dir", str(refine_output),
            "--refine-close-clusters", "0.02",
            "--min-split", "0.005",
            "--max-lump", "0.02"
        ], capture_output=True, text=True, timeout=300)

        assert result.returncode == 0, f"Refine with close clusters failed: {result.stderr}"

        refined_clusters_dir = refine_output / "latest"
        refined_clusters = len(list(refined_clusters_dir.glob("cluster_*.fasta")))
        print(f"Refined clusters: {refined_clusters}")

        # Close cluster refinement should merge some clusters
        # (exact number depends on data, but should be <= initial)
        assert refined_clusters <= initial_clusters

        # Check summary report mentions close cluster refinement
        summary_file = refined_clusters_dir / "refine_summary.txt"
        with open(summary_file) as f:
            summary = f.read()

        # New two-pass mode uses different naming
        assert "Pass 2" in summary or "Stage 2" in summary
        assert "Status: APPLIED" in summary or "Status: SKIPPED" in summary or "APPLIED" in summary

    def test_chained_refinement(self, russula_50_fasta, temp_work_dir):
        """Test running refinement multiple times (chained)."""
        if not russula_50_fasta.exists():
            pytest.skip("Russula 50 dataset not found")

        # Step 1: Initial decompose
        decompose_output = temp_work_dir / "decompose_out"
        subprocess.run([
            "gaphack-decompose",
            str(russula_50_fasta),
            "-o", str(decompose_output),
            "--max-clusters", "12",
        ], capture_output=True, text=True, check=True)

        clusters_dir = decompose_output / "clusters" / "latest"

        # Step 2: First refinement (conservative)
        refine1_output = temp_work_dir / "refine1"
        subprocess.run([
            "gaphack-refine",
            "--input-dir", str(clusters_dir),
            "--output-dir", str(refine1_output),
            "--refine-close-clusters", "0.015",
        ], capture_output=True, text=True, check=True, timeout=300)

        refine1_clusters = refine1_output / "latest"
        round1_count = len(list(refine1_clusters.glob("cluster_*.fasta")))
        print(f"Round 1: {round1_count} clusters")

        # Step 3: Second refinement (more aggressive)
        refine2_output = temp_work_dir / "refine2"
        subprocess.run([
            "gaphack-refine",
            "--input-dir", str(refine1_clusters),
            "--output-dir", str(refine2_output),
            "--refine-close-clusters", "0.025",
        ], capture_output=True, text=True, check=True, timeout=300)

        refine2_clusters = refine2_output / "latest"
        round2_count = len(list(refine2_clusters.glob("cluster_*.fasta")))
        print(f"Round 2: {round2_count} clusters")

        # More aggressive refinement should produce fewer or equal clusters
        assert round2_count <= round1_count

        # Verify MECE property still holds after chaining
        sequence_assignments = {}
        for cluster_file in refine2_clusters.glob("cluster_*.fasta"):
            cluster_id = cluster_file.stem
            for record in SeqIO.parse(cluster_file, "fasta"):
                if record.id in sequence_assignments:
                    pytest.fail(f"After chained refinement, sequence {record.id} in multiple clusters")
                sequence_assignments[record.id] = cluster_id

        print(f"Chained refinement: MECE property verified")


@pytest.mark.integration
class TestRefineWithVsearch:
    """Test refining output from external clustering tools (simulated)."""

    def test_refine_vsearch_format_clusters(self, temp_work_dir):
        """Test refining vsearch-style cluster naming."""
        # Create simulated vsearch output
        vsearch_dir = temp_work_dir / "vsearch_clusters"
        vsearch_dir.mkdir()

        from Bio.Seq import Seq
        from Bio.SeqRecord import SeqRecord

        # Create vsearch-style named clusters
        clusters = {
            "vsearch_cluster_0": [
                SeqRecord(Seq("ACGTACGTACGTACGT"), id="seq1", description=""),
                SeqRecord(Seq("ACGTACGTACGTACGG"), id="seq2", description=""),
                SeqRecord(Seq("ACGTACGTACGTACGA"), id="seq3", description=""),
            ],
            "vsearch_cluster_1": [
                SeqRecord(Seq("TTTTTTTTTTTTTTTT"), id="seq4", description=""),
                SeqRecord(Seq("TTTTTTTTTTTTTTGG"), id="seq5", description=""),
            ],
        }

        for cluster_name, records in clusters.items():
            cluster_file = vsearch_dir / f"{cluster_name}.fasta"
            with open(cluster_file, 'w') as f:
                SeqIO.write(records, f, "fasta")

        # Run refine on vsearch output
        refine_output = temp_work_dir / "refined"
        result = subprocess.run([
            "gaphack-refine",
            "--input-dir", str(vsearch_dir),
            "--output-dir", str(refine_output),
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Refine failed: {result.stderr}"

        # Check output
        refined_dir = refine_output / "latest"
        assert (refined_dir / "cluster_00001.fasta").exists()
        assert (refined_dir / "cluster_00002.fasta").exists()

        # Check mapping preserves original vsearch names
        mapping_file = refined_dir / "cluster_mapping.txt"
        with open(mapping_file) as f:
            mapping_content = f.read()

        assert "vsearch_cluster_0" in mapping_content
        assert "vsearch_cluster_1" in mapping_content

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
            "--close-threshold", "0.02",
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

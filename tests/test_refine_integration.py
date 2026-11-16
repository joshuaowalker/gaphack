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


@pytest.mark.integration
class TestRefinementConvergence:
    """Refinement must stop as soon as an iteration leaves the clustering unchanged."""

    def test_stops_before_iteration_limit_and_reuses_medoids(self):
        from unittest.mock import patch
        from gaphack import cluster_refinement
        from gaphack.cluster_refinement import RefinementConfig, refine_clusters
        from gaphack.utils import load_sequences_from_fasta

        fasta = Path(__file__).parent.parent / "examples" / "data" / "collybia_nuda_test.fasta"
        sequences, headers, _ = load_sequences_from_fasta(str(fasta))
        half = len(headers) // 2
        start = {"start_a": headers[:half], "start_b": headers[half:]}

        config = RefinementConfig(close_threshold=0.02, max_iterations=10)
        with patch.object(cluster_refinement, "ClusterGraph",
                          wraps=cluster_refinement.ClusterGraph) as graph_spy:
            refined, info = refine_clusters(
                all_clusters=start, sequences=sequences, headers=headers,
                min_split=0.005, max_lump=0.02, target_percentile=95, config=config,
            )

        # Every sequence is still in exactly one cluster
        assert sorted(h for members in refined.values() for h in members) == sorted(headers)

        # This dataset settles on cluster sizes for which sklearn's AMI of two identical
        # partitions is 1.0000000000000002; the loop used to miss that and run all 10 iterations.
        assert sorted((len(m) for m in refined.values()), reverse=True) == [65, 20, 3, 1, 1, 1]
        assert info.summary_stats["convergence_reason"] == "ami_convergence"
        assert info.summary_stats["iterations"] < config.max_iterations

        # One proximity graph per iteration, all sharing one medoid cache
        assert graph_spy.call_count == info.summary_stats["iterations"]
        medoid_caches = [call.kwargs["medoid_computation_cache"] for call in graph_spy.call_args_list]
        assert all(cache is medoid_caches[0] for cache in medoid_caches)
        assert len(medoid_caches[0]) > 0

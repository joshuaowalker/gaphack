"""Tests for Phase 4: Staged Refinement capability.

This test suite validates:
1. Applying conflict resolution using gaphack-refine CLI
2. Applying close cluster refinement using gaphack-refine CLI
3. Chaining refinement stages across decompose and refine
4. Tracking refinement history in state
5. Verifying MECE property after refinement

NOTE: These tests now use the gaphack-refine CLI to test refinement functionality
that has been separated from gaphack-decompose.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from gaphack.decompose import DecomposeClustering
from gaphack.state import DecomposeState
from test_phase4_integration import CLIRunner


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_fasta(temp_output_dir):
    """Create sample FASTA file with sequences that will create conflicts.

    Creates 15 sequences:
    - Group A (5 sequences): very similar (98% identity)
    - Group B (5 sequences): very similar (98% identity)
    - Group C (5 sequences): intermediate between A and B (90% identity to both)

    This creates a scenario where Group C sequences might get assigned to both
    Group A and B clusters, creating conflicts.
    """
    # Base sequences
    base_seq_a = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 10  # 400bp
    base_seq_b = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA" * 10  # 400bp
    base_seq_c = "ACGTTGCAACGTTGCAACGTTGCAACGTTGCAACGTTGCA" * 10  # 400bp (intermediate)

    records = []

    # Group A sequences (very similar to each other)
    for i in range(5):
        seq = base_seq_a[:400]
        if i > 0:
            # Introduce small variations (2%)
            seq_list = list(seq)
            for j in range(0, len(seq_list), 50):
                if j + i < len(seq_list):
                    seq_list[j + i] = 'T' if seq_list[j + i] == 'A' else 'A'
            seq = ''.join(seq_list)
        records.append(SeqRecord(Seq(seq), id=f"seq_a_{i+1}", description=""))

    # Group B sequences (very similar to each other)
    for i in range(5):
        seq = base_seq_b[:400]
        if i > 0:
            # Introduce small variations (2%)
            seq_list = list(seq)
            for j in range(0, len(seq_list), 50):
                if j + i < len(seq_list):
                    seq_list[j + i] = 'C' if seq_list[j + i] == 'T' else 'T'
            seq = ''.join(seq_list)
        records.append(SeqRecord(Seq(seq), id=f"seq_b_{i+1}", description=""))

    # Group C sequences (intermediate - might cause conflicts)
    for i in range(5):
        seq = base_seq_c[:400]
        if i > 0:
            # Introduce small variations
            seq_list = list(seq)
            for j in range(0, len(seq_list), 50):
                if j + i < len(seq_list):
                    seq_list[j + i] = 'G' if seq_list[j + i] != 'G' else 'C'
            seq = ''.join(seq_list)
        records.append(SeqRecord(Seq(seq), id=f"seq_c_{i+1}", description=""))

    fasta_path = temp_output_dir / "sample_sequences.fasta"
    with open(fasta_path, 'w') as f:
        SeqIO.write(records, f, "fasta")

    return fasta_path


def test_apply_conflict_resolution_after_initial_clustering(sample_fasta, temp_output_dir):
    """Test applying conflict resolution to completed initial clustering.

    Scenario:
    1. Run gaphack-decompose (initial clustering only)
    2. Run gaphack-refine for conflict resolution
    3. Verify conflicts are resolved
    4. Verify MECE property
    """
    output_dir = temp_output_dir / "test_conflict_resolution"
    output_dir.mkdir()

    # Step 1: Run gaphack-decompose (initial clustering only)
    decomposer = DecomposeClustering(
        min_split=0.01,
        max_lump=0.05,
        target_percentile=95,
        show_progress=False
    )

    results = decomposer.decompose(
        input_fasta=str(sample_fasta),
        output_dir=str(output_dir)
    )

    # Verify initial clustering completed
    initial_clusters_dir = output_dir / "work" / "initial"
    assert initial_clusters_dir.exists(), "Initial clusters directory should exist"
    initial_files = list(initial_clusters_dir.glob("cluster_*.fasta"))
    assert len(initial_files) > 0, "Initial cluster files should exist"

    # Step 2: Run gaphack-refine for conflict resolution
    refine_output = temp_output_dir / "refined_conflict_resolution"
    result = CLIRunner.run_gaphack_refine(
        input_dir=initial_clusters_dir,
        output_dir=refine_output,
        no_timestamp=True  # Write directly to output_dir
    )

    # Step 3: Verify refinement succeeded
    assert result['returncode'] == 0, f"Refine should succeed. stderr: {result['stderr']}"

    # Check that refined clusters exist
    refined_files = list(refine_output.glob("cluster_*.fasta"))
    assert len(refined_files) > 0, "Refined cluster files should exist"

    # Step 4: Verify MECE property (no sequence appears in multiple clusters)
    all_sequences = set()
    for cluster_file in refined_files:
        cluster_sequences = {record.id for record in SeqIO.parse(cluster_file, "fasta")}
        overlap = all_sequences & cluster_sequences
        assert len(overlap) == 0, f"MECE violation: sequences {overlap} appear in multiple clusters"
        all_sequences.update(cluster_sequences)


def test_apply_close_cluster_refinement_after_initial_clustering(sample_fasta, temp_output_dir):
    """Test applying close cluster refinement to completed clustering.

    Scenario:
    1. Run gaphack-decompose (initial clustering only)
    2. Run gaphack-refine with close cluster refinement
    3. Verify clusters are refined
    4. Verify MECE property
    """
    output_dir = temp_output_dir / "test_close_refinement"
    output_dir.mkdir()

    # Step 1: Run gaphack-decompose
    decomposer = DecomposeClustering(
        min_split=0.01,
        max_lump=0.05,
        target_percentile=95,
        show_progress=False
    )

    results = decomposer.decompose(
        input_fasta=str(sample_fasta),
        output_dir=str(output_dir)
    )

    # Verify initial clustering completed
    initial_clusters_dir = output_dir / "work" / "initial"
    assert initial_clusters_dir.exists(), "Initial clusters directory should exist"
    initial_files = list(initial_clusters_dir.glob("cluster_*.fasta"))
    assert len(initial_files) > 0, "Initial cluster files should exist"
    clusters_before = len(initial_files)

    # Step 2: Run gaphack-refine with close cluster refinement
    refine_output = temp_output_dir / "refined_close_clusters"
    result = CLIRunner.run_gaphack_refine(
        input_dir=initial_clusters_dir,
        output_dir=refine_output,
        refine_close_clusters=0.05,
        no_timestamp=True
    )

    # Step 3: Verify refinement succeeded
    assert result['returncode'] == 0, f"Refine should succeed. stderr: {result['stderr']}"

    # Check that refined clusters exist
    refined_files = list(refine_output.glob("cluster_*.fasta"))
    assert len(refined_files) > 0, "Refined cluster files should exist"

    # Step 4: Verify MECE property
    all_sequences = set()
    for cluster_file in refined_files:
        cluster_sequences = {record.id for record in SeqIO.parse(cluster_file, "fasta")}
        overlap = all_sequences & cluster_sequences
        assert len(overlap) == 0, f"MECE violation: sequences {overlap} appear in multiple clusters"
        all_sequences.update(cluster_sequences)


def test_chained_refinement_stages(sample_fasta, temp_output_dir):
    """Test chaining multiple refinement stages.

    Scenario:
    1. Run gaphack-decompose (initial clustering)
    2. Run gaphack-refine with conflict resolution
    3. Run gaphack-refine again with close cluster refinement on deconflicted results
    4. Verify MECE property is maintained throughout
    """
    output_dir = temp_output_dir / "test_chained_refinement"
    output_dir.mkdir()

    # Step 1: Run gaphack-decompose
    decomposer = DecomposeClustering(
        min_split=0.01,
        max_lump=0.05,
        target_percentile=95,
        show_progress=False
    )

    decomposer.decompose(
        input_fasta=str(sample_fasta),
        output_dir=str(output_dir)
    )

    # Verify initial clustering
    initial_clusters_dir = output_dir / "work" / "initial"
    assert initial_clusters_dir.exists()
    initial_files = list(initial_clusters_dir.glob("cluster_*.fasta"))
    assert len(initial_files) > 0

    # Step 2: Apply conflict resolution with gaphack-refine
    refine_output_1 = temp_output_dir / "refined_stage1"
    result_1 = CLIRunner.run_gaphack_refine(
        input_dir=initial_clusters_dir,
        output_dir=refine_output_1,
        no_timestamp=True
    )
    assert result_1['returncode'] == 0, f"First refinement should succeed. stderr: {result_1['stderr']}"

    # Verify MECE after conflict resolution
    all_sequences_1 = set()
    refined_files_1 = list(refine_output_1.glob("cluster_*.fasta"))
    assert len(refined_files_1) > 0
    for cluster_file in refined_files_1:
        cluster_sequences = {record.id for record in SeqIO.parse(cluster_file, "fasta")}
        overlap = all_sequences_1 & cluster_sequences
        assert len(overlap) == 0, f"MECE violation after stage 1: {overlap}"
        all_sequences_1.update(cluster_sequences)

    # Step 3: Apply close cluster refinement on deconflicted results
    refine_output_2 = temp_output_dir / "refined_stage2"
    result_2 = CLIRunner.run_gaphack_refine(
        input_dir=refine_output_1,
        output_dir=refine_output_2,
        refine_close_clusters=0.05,
        no_timestamp=True
    )
    assert result_2['returncode'] == 0, f"Second refinement should succeed. stderr: {result_2['stderr']}"

    # Verify MECE after close cluster refinement
    all_sequences_2 = set()
    refined_files_2 = list(refine_output_2.glob("cluster_*.fasta"))
    assert len(refined_files_2) > 0
    for cluster_file in refined_files_2:
        cluster_sequences = {record.id for record in SeqIO.parse(cluster_file, "fasta")}
        overlap = all_sequences_2 & cluster_sequences
        assert len(overlap) == 0, f"MECE violation after stage 2: {overlap}"
        all_sequences_2.update(cluster_sequences)

    # Verify all sequences are present in final output
    assert all_sequences_1 == all_sequences_2, "All sequences should be preserved across refinement stages"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
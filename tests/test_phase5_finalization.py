"""Tests for Phase 5: Finalization capability.

This test suite validates:
1. Creating final numbered cluster output
2. Renumbering clusters by size
3. Marking state as finalized
4. Optional cleanup of intermediate files
5. Handling already-finalized directories
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from gaphack.decompose import DecomposeClustering
from gaphack.resume import resume_decompose, finalize_decompose
from gaphack.state import DecomposeState


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_fasta(temp_output_dir):
    """Create sample FASTA file with sequences for clustering.

    Creates 30 sequences in 3 groups (10 each) with high similarity within groups.
    """
    # Base sequences for 3 groups
    base_seq_a = "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" * 10  # 400bp
    base_seq_b = "TGCATGCATGCATGCATGCATGCATGCATGCATGCATGCA" * 10  # 400bp
    base_seq_c = "GGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCCGGCC" * 10  # 400bp

    records = []

    # Group A (10 sequences)
    for i in range(10):
        seq = base_seq_a[:400]
        if i > 0:
            seq_list = list(seq)
            for j in range(0, len(seq_list), 50):
                if j + i < len(seq_list):
                    seq_list[j + i] = 'T' if seq_list[j + i] == 'A' else 'A'
            seq = ''.join(seq_list)
        records.append(SeqRecord(Seq(seq), id=f"seq_a_{i+1:02d}", description=""))

    # Group B (10 sequences)
    for i in range(10):
        seq = base_seq_b[:400]
        if i > 0:
            seq_list = list(seq)
            for j in range(0, len(seq_list), 50):
                if j + i < len(seq_list):
                    seq_list[j + i] = 'C' if seq_list[j + i] == 'T' else 'T'
            seq = ''.join(seq_list)
        records.append(SeqRecord(Seq(seq), id=f"seq_b_{i+1:02d}", description=""))

    # Group C (10 sequences)
    for i in range(10):
        seq = base_seq_c[:400]
        if i > 0:
            seq_list = list(seq)
            for j in range(0, len(seq_list), 50):
                if j + i < len(seq_list):
                    seq_list[j + i] = 'A' if seq_list[j + i] == 'G' else 'G'
            seq = ''.join(seq_list)
        records.append(SeqRecord(Seq(seq), id=f"seq_c_{i+1:02d}", description=""))

    fasta_path = temp_output_dir / "sample_sequences.fasta"
    with open(fasta_path, 'w') as f:
        SeqIO.write(records, f, "fasta")

    return fasta_path


def test_finalize_from_initial_clustering(sample_fasta, temp_output_dir):
    """Test finalizing directly from initial clustering (no refinement).

    Scenario:
    1. Run initial clustering to completion
    2. Finalize to create numbered output
    3. Verify final files exist and are correctly numbered
    4. Verify state is marked as finalized
    """
    output_dir = temp_output_dir / "test_finalize_initial"
    output_dir.mkdir()

    # Step 1: Run initial clustering
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

    # Verify initial files exist
    initial_files = list((output_dir / "work/initial").glob("*.fasta"))
    assert len(initial_files) > 0, "Initial cluster files should exist"
    num_clusters = len(initial_files)

    # Step 2: Finalize
    finalize_decompose(output_dir=str(output_dir), cleanup=False)

    # Step 3: Verify final numbered files exist in clusters/latest
    latest_dir = output_dir / "clusters/latest"
    if latest_dir.exists():
        final_files = sorted(latest_dir.glob("*.fasta"))
        # Filter out unassigned.fasta if it exists
        final_files = [f for f in final_files if f.name != "unassigned.fasta"]
        assert len(final_files) == num_clusters, f"Expected {num_clusters} final files, got {len(final_files)}"
    else:
        # Fallback: check root output directory for cluster files
        final_files = sorted(output_dir.glob("cluster_*.fasta"))
        assert len(final_files) == num_clusters, f"Expected {num_clusters} final files, got {len(final_files)}"

    # Verify clusters are ordered by size (largest first)
    cluster_sizes = []
    for final_file in final_files:
        records = list(SeqIO.parse(final_file, "fasta"))
        cluster_sizes.append(len(records))

    # Check that sizes are non-increasing
    for i in range(len(cluster_sizes) - 1):
        assert cluster_sizes[i] >= cluster_sizes[i+1], \
            f"Cluster sizes should be non-increasing: {cluster_sizes}"

    # Step 4: Verify state is marked as finalized
    state = DecomposeState.load(output_dir)
    assert state.finalized.completed, "State should be marked as finalized"
    assert state.stage == "finalized", "Stage should be 'finalized'"
    assert state.finalized.source_stage == "initial", "Source stage should be 'initial'"
    assert state.finalized.total_clusters == num_clusters, "Total clusters should match"
    assert state.finalized.total_sequences > 0, "Total sequences should be recorded"

    # Verify initial files are NOT removed (cleanup=False)
    initial_files_after = list((output_dir / "work/initial").glob("*.fasta"))
    assert len(initial_files_after) == len(initial_files), "Initial files should still exist"


def test_finalize_after_refinement(sample_fasta, temp_output_dir):
    """Test finalizing after applying refinement stages.

    Scenario:
    1. Run gaphack-decompose (initial clustering)
    2. Run gaphack-refine (conflict resolution)
    3. Verify refined clusters are properly numbered and complete
    4. Verify MECE property
    """
    output_dir = temp_output_dir / "test_finalize_refinement"
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

    # Get initial clusters
    initial_clusters_dir = output_dir / "work" / "initial"
    assert initial_clusters_dir.exists(), "Initial clusters directory should exist"

    # Step 2: Run gaphack-refine
    from test_phase4_integration import CLIRunner

    refine_output = temp_output_dir / "refined_final"
    result = CLIRunner.run_gaphack_refine(
        input_dir=initial_clusters_dir,
        output_dir=refine_output,
        no_timestamp=True,
        renumber=True  # Ensure clusters are renumbered by size
    )

    # Step 3: Verify refinement succeeded
    assert result['returncode'] == 0, f"Refinement should succeed. stderr: {result['stderr']}"

    # Check that refined/final cluster files exist
    final_files = sorted(refine_output.glob("cluster_*.fasta"))
    assert len(final_files) > 0, "Final cluster files should exist"

    # Step 4: Verify clusters are ordered by size (largest first)
    cluster_sizes = []
    for final_file in final_files:
        records = list(SeqIO.parse(final_file, "fasta"))
        cluster_sizes.append(len(records))

    # Check that sizes are non-increasing (largest first)
    for i in range(len(cluster_sizes) - 1):
        assert cluster_sizes[i] >= cluster_sizes[i+1], \
            f"Cluster sizes should be non-increasing: {cluster_sizes}"

    # Step 5: Verify MECE property
    all_sequences = set()
    for cluster_file in final_files:
        cluster_sequences = {record.id for record in SeqIO.parse(cluster_file, "fasta")}
        overlap = all_sequences & cluster_sequences
        assert len(overlap) == 0, f"MECE violation: sequences {overlap} appear in multiple clusters"
        all_sequences.update(cluster_sequences)


def test_finalize_with_cleanup(sample_fasta, temp_output_dir):
    """Test finalization with cleanup of intermediate files.

    Scenario:
    1. Run initial clustering
    2. Apply refinement (creates multiple stage files)
    3. Finalize with cleanup=True
    4. Verify intermediate files are removed
    5. Verify source stage files are retained
    """
    output_dir = temp_output_dir / "test_finalize_cleanup"
    output_dir.mkdir()

    # Step 1: Initial clustering
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

    initial_files_before = list((output_dir / "work/initial").glob("*.fasta"))
    assert len(initial_files_before) > 0

    # Step 2: Apply conflict resolution
    resume_decompose(output_dir=output_dir, resolve_conflicts=True)

    # Check if deconflicted files were created
    deconflicted_dir = output_dir / "work/deconflicted"
    deconflicted_files_before = list(deconflicted_dir.glob("*.fasta")) if deconflicted_dir.exists() else []

    # Step 3: Finalize with cleanup
    finalize_decompose(output_dir=str(output_dir), cleanup=True)

    # Step 4: Verify cleanup
    initial_files_after = list((output_dir / "work/initial").glob("*.fasta")) if (output_dir / "work/initial").exists() else []

    # If we finalized from deconflicted stage, initial files should be removed
    state = DecomposeState.load(output_dir)
    if state.finalized.source_stage == "deconflicted":
        assert len(initial_files_after) == 0, "Initial files should be removed when cleanup=True"
        # Deconflicted files should be retained (they're the source)
        deconflicted_files_after = list(deconflicted_dir.glob("*.fasta")) if deconflicted_dir.exists() else []
        if len(deconflicted_files_before) > 0:
            assert len(deconflicted_files_after) > 0, "Source stage files should be retained"
    else:
        # If finalized from initial, initial files should be retained
        assert len(initial_files_after) > 0, "Source stage files should be retained"

    # Step 5: Verify final files exist
    latest_dir = output_dir / "clusters/latest"
    if latest_dir.exists():
        final_files = [f for f in latest_dir.glob("*.fasta") if f.name != "unassigned.fasta"]
    else:
        final_files = list(output_dir.glob("cluster_*.fasta"))
    assert len(final_files) > 0, "Final cluster files should exist"


def test_finalize_already_finalized(sample_fasta, temp_output_dir):
    """Test attempting to finalize an already-finalized directory.

    Scenario:
    1. Run and finalize
    2. Attempt to finalize again
    3. Verify no errors and state unchanged
    """
    output_dir = temp_output_dir / "test_finalize_twice"
    output_dir.mkdir()

    # Step 1: Run and finalize
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

    finalize_decompose(output_dir=str(output_dir), cleanup=False)

    state_first = DecomposeState.load(output_dir)
    assert state_first.finalized.completed

    # Count final files
    latest_dir = output_dir / "clusters/latest"
    if latest_dir.exists():
        final_files_first = [f for f in latest_dir.glob("*.fasta") if f.name != "unassigned.fasta"]
    else:
        final_files_first = list(output_dir.glob("cluster_*.fasta"))
    count_first = len(final_files_first)

    # Step 2: Attempt to finalize again
    finalize_decompose(output_dir=str(output_dir), cleanup=False)

    # Step 3: Verify state unchanged
    state_second = DecomposeState.load(output_dir)
    assert state_second.finalized.completed
    assert state_second.finalized.total_clusters == state_first.finalized.total_clusters

    # Verify file count unchanged
    if latest_dir.exists():
        final_files_second = [f for f in latest_dir.glob("*.fasta") if f.name != "unassigned.fasta"]
    else:
        final_files_second = list(output_dir.glob("cluster_*.fasta"))
    assert len(final_files_second) == count_first, "File count should be unchanged"


def test_finalize_incomplete_clustering_fails(sample_fasta, temp_output_dir):
    """Test that finalization fails if initial clustering is incomplete.

    Scenario:
    1. Run limited initial clustering (incomplete)
    2. Attempt to finalize
    3. Verify error is raised
    """
    output_dir = temp_output_dir / "test_finalize_incomplete"
    output_dir.mkdir()

    # Step 1: Run limited clustering (will not complete)
    decomposer = DecomposeClustering(
        min_split=0.01,
        max_lump=0.05,
        target_percentile=95,
        show_progress=False
    )

    decomposer.decompose(
        input_fasta=str(sample_fasta),
        output_dir=str(output_dir),
        max_clusters=2  # Limit to 2 clusters (incomplete)
    )

    state = DecomposeState.load(output_dir)
    assert not state.initial_clustering.completed, "Initial clustering should be incomplete"

    # Step 2: Attempt to finalize should raise error
    with pytest.raises(ValueError, match="Cannot finalize: initial clustering is not complete"):
        finalize_decompose(output_dir=str(output_dir), cleanup=False)


def test_finalize_preserves_sequence_content(sample_fasta, temp_output_dir):
    """Test that finalization preserves all sequence content correctly.

    Scenario:
    1. Run clustering
    2. Finalize
    3. Verify all sequences from INPUT file appear in final files
    4. Verify sequence content is unchanged

    Note: Initial stage files contain hash IDs, final files contain original headers.
    So we compare against the original input file, not the initial stage files.
    """
    output_dir = temp_output_dir / "test_finalize_content"
    output_dir.mkdir()

    # Step 1: Run clustering
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

    # Read all sequences from ORIGINAL INPUT (not initial stage files)
    input_sequences = {}
    for record in SeqIO.parse(sample_fasta, "fasta"):
        input_sequences[record.id] = str(record.seq)

    input_count = len(input_sequences)

    # Step 2: Finalize
    finalize_decompose(output_dir=str(output_dir), cleanup=False)

    # Step 3: Read all sequences from final files
    final_sequences = {}
    latest_dir = output_dir / "clusters/latest"
    if latest_dir.exists():
        for final_file in latest_dir.glob("*.fasta"):
            if final_file.name != "unassigned.fasta":
                for record in SeqIO.parse(final_file, "fasta"):
                    final_sequences[record.id] = str(record.seq)

        # Also check unassigned if it exists
        unassigned_file = latest_dir / "unassigned.fasta"
        if unassigned_file.exists():
            for record in SeqIO.parse(unassigned_file, "fasta"):
                final_sequences[record.id] = str(record.seq)
    else:
        # Fallback: check root directory
        for final_file in output_dir.glob("cluster_*.fasta"):
            for record in SeqIO.parse(final_file, "fasta"):
                final_sequences[record.id] = str(record.seq)

        unassigned_file = output_dir / "unassigned.fasta"
        if unassigned_file.exists():
            for record in SeqIO.parse(unassigned_file, "fasta"):
                final_sequences[record.id] = str(record.seq)

    # Step 4: Verify all sequences are present and unchanged
    # Note: final might have more if there were duplicates in input
    assert len(final_sequences) >= input_count, \
        f"Expected at least {input_count} sequences, got {len(final_sequences)}"

    # Check that all input sequences appear in final output
    for seq_id, seq in input_sequences.items():
        assert seq_id in final_sequences, f"Sequence {seq_id} missing from final output"
        assert final_sequences[seq_id] == seq, f"Sequence {seq_id} content changed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
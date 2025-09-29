"""Tests for Phase 4: Staged Refinement capability.

This test suite validates:
1. Applying conflict resolution to resumed clustering
2. Applying close cluster refinement to resumed clustering
3. Chaining refinement stages
4. Tracking refinement history in state
5. Stage completion flags
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from gaphack.decompose import DecomposeClustering, resume_decompose
from gaphack.state import DecomposeState


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
    1. Run initial clustering without conflict resolution
    2. Resume with resolve_conflicts=True
    3. Verify conflicts are resolved
    4. Verify state tracks conflict resolution completion
    """
    output_dir = temp_output_dir / "test_conflict_resolution"
    output_dir.mkdir()

    # Step 1: Initial clustering without conflict resolution
    # Run to completion (no max_clusters limit)
    decomposer = DecomposeClustering(
        min_split=0.01,
        max_lump=0.05,
        target_percentile=95,
        resolve_conflicts=False,  # Explicitly disable
        show_progress=False
    )

    results = decomposer.decompose(
        input_fasta=str(sample_fasta),
        output_dir=str(output_dir)
    )

    # Verify state shows initial clustering complete
    state = DecomposeState.load(output_dir)
    assert state.initial_clustering.completed, "Initial clustering should be marked complete"
    assert state.stage == "initial_clustering"
    assert not state.conflict_resolution.completed, "Conflict resolution should not be complete"

    # Step 2: Resume with conflict resolution
    results_after = resume_decompose(
        output_dir=output_dir,
        resolve_conflicts=True
    )

    # Step 3: Verify state after conflict resolution
    state_after = DecomposeState.load(output_dir)
    assert state_after.conflict_resolution.completed, "Conflict resolution should be marked complete"
    assert state_after.stage == "conflict_resolution"
    assert state_after.conflict_resolution.conflicts_after == 0, "All conflicts should be resolved"

    # If there were conflicts, deconflicted files should exist
    # If there were no conflicts, files aren't created (pattern stays as initial)
    if state_after.conflict_resolution.conflicts_before > 0:
        deconflicted_files = list(output_dir.glob("deconflicted.cluster_*.fasta"))
        assert len(deconflicted_files) > 0, "Deconflicted cluster files should exist when conflicts resolved"
        assert state_after.conflict_resolution.cluster_file_pattern == "deconflicted.cluster_*.fasta"
    else:
        # No conflicts - pattern should remain as initial clustering
        assert state_after.conflict_resolution.cluster_file_pattern == "initial.cluster_*.fasta"


def test_apply_close_cluster_refinement_after_initial_clustering(sample_fasta, temp_output_dir):
    """Test applying close cluster refinement to completed clustering.

    Scenario:
    1. Run initial clustering without refinement
    2. Resume with refine_close_clusters parameter
    3. Verify clusters are refined
    4. Verify state tracks refinement completion
    """
    output_dir = temp_output_dir / "test_close_refinement"
    output_dir.mkdir()

    # Step 1: Initial clustering without refinement
    decomposer = DecomposeClustering(
        min_split=0.01,
        max_lump=0.05,
        target_percentile=95,
        refine_close_clusters=False,
        show_progress=False
    )

    results = decomposer.decompose(
        input_fasta=str(sample_fasta),
        output_dir=str(output_dir)
    )

    # Verify state
    state = DecomposeState.load(output_dir)
    assert state.initial_clustering.completed
    clusters_before = state.initial_clustering.total_clusters

    # Step 2: Resume with close cluster refinement
    results_after = resume_decompose(
        output_dir=output_dir,
        refine_close_clusters=0.05
    )

    # Step 3: Verify state after refinement
    state_after = DecomposeState.load(output_dir)
    assert state_after.close_cluster_refinement.completed, "Refinement should be marked complete"
    assert state_after.stage == "close_cluster_refinement"
    assert state_after.close_cluster_refinement.threshold == 0.05

    # Verify refined FASTA files exist
    refined_files = list(output_dir.glob("refined.cluster_*.fasta"))
    assert len(refined_files) > 0, "Refined cluster files should exist"

    # Verify pattern is updated
    assert state_after.close_cluster_refinement.cluster_file_pattern == "refined.cluster_*.fasta"

    # Verify refinement history
    assert len(state_after.close_cluster_refinement.refinement_history) == 1
    history_entry = state_after.close_cluster_refinement.refinement_history[0]
    assert history_entry['threshold'] == 0.05
    assert 'timestamp' in history_entry


def test_chained_refinement_stages(sample_fasta, temp_output_dir):
    """Test chaining multiple refinement stages.

    Scenario:
    1. Run initial clustering
    2. Apply conflict resolution
    3. Apply close cluster refinement (should chain on deconflicted results)
    4. Verify each stage updates correctly
    """
    output_dir = temp_output_dir / "test_chained_refinement"
    output_dir.mkdir()

    # Step 1: Initial clustering
    decomposer = DecomposeClustering(
        min_split=0.01,
        max_lump=0.05,
        target_percentile=95,
        resolve_conflicts=False,
        refine_close_clusters=False,
        show_progress=False
    )

    decomposer.decompose(
        input_fasta=str(sample_fasta),
        output_dir=str(output_dir)
    )

    state_1 = DecomposeState.load(output_dir)
    clusters_initial = state_1.initial_clustering.total_clusters

    # Step 2: Apply conflict resolution
    resume_decompose(output_dir=output_dir, resolve_conflicts=True)

    state_2 = DecomposeState.load(output_dir)
    assert state_2.conflict_resolution.completed
    assert state_2.stage == "conflict_resolution"
    clusters_deconflicted = state_2.conflict_resolution.total_clusters

    # Step 3: Apply close cluster refinement (should use deconflicted results)
    resume_decompose(output_dir=output_dir, refine_close_clusters=0.05)

    state_3 = DecomposeState.load(output_dir)
    assert state_3.close_cluster_refinement.completed
    assert state_3.stage == "close_cluster_refinement"

    # Verify clusters_before for refinement stage matches deconflicted output
    assert state_3.close_cluster_refinement.clusters_before == clusters_deconflicted

    # Verify initial stage files exist
    assert len(list(output_dir.glob("initial.cluster_*.fasta"))) > 0

    # Deconflicted files may not exist if no conflicts found
    # Refined files may not exist if clusters were already optimal
    # The important thing is that stages completed successfully


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
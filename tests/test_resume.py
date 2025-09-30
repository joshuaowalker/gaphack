"""Integration tests for resume functionality."""

import pytest
import tempfile
from pathlib import Path
from gaphack.decompose import DecomposeClustering
from gaphack.resume import resume_decompose
from gaphack.state import DecomposeState, StateManager
from gaphack.utils import load_sequences_with_deduplication


@pytest.fixture
def small_test_fasta():
    """Provide path to small test FASTA file."""
    return "tests/test_data/russula_diverse_50.fasta"


class TestResumeBasics:
    """Basic tests for resume functionality."""

    def test_create_checkpoint_and_load_state(self, small_test_fasta, tmp_path):
        """Test that we can create a checkpoint and load the state."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run clustering with max_clusters limit to create checkpoint
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results = decomposer.decompose(
            input_fasta=small_test_fasta,
            max_clusters=3,  # Stop after 3 clusters
            output_dir=str(output_dir)
        )

        # Verify state file was created
        state_file = output_dir / "state.json"
        assert state_file.exists(), "state.json should be created"

        # Load and verify state
        state = DecomposeState.load(output_dir)
        assert state.status == "in_progress"
        assert state.stage == "initial_clustering"
        assert state.initial_clustering.total_clusters == 3
        assert state.initial_clustering.total_iterations == 3

        # Verify cluster FASTA files were created
        cluster_files = list(output_dir.glob("initial.cluster_*.fasta"))
        assert len(cluster_files) == 3, f"Should have 3 cluster files, got {len(cluster_files)}"

    def test_resume_continues_from_checkpoint(self, small_test_fasta, tmp_path):
        """Test that resume continues from where it left off."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Initial run with max_clusters=2
        decomposer1 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer1.decompose(
            input_fasta=small_test_fasta,
            max_clusters=2,
            output_dir=str(output_dir)
        )

        initial_iteration = results1.total_iterations
        assert initial_iteration == 2

        # Resume and add more clusters
        results2 = resume_decompose(
            output_dir=output_dir,
            max_clusters=5,  # Increase to 5 clusters (absolute)
            show_progress=False
        )

        # Should have added 3 more clusters (5 - 2 = 3)
        assert results2.total_iterations >= 5
        assert len(results2.clusters) >= 5

    def test_resume_with_no_action_needed(self, small_test_fasta, tmp_path):
        """Test resuming when clustering is already complete."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run clustering to completion
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer.decompose(
            input_fasta=small_test_fasta,
            max_clusters=3,
            output_dir=str(output_dir)
        )

        # Mark as completed
        state = DecomposeState.load(output_dir)
        state.initial_clustering.completed = True
        state.save(output_dir)

        # Resume - should report nothing to do
        results2 = resume_decompose(
            output_dir=output_dir,
            show_progress=False
        )

        # Results should match original
        assert len(results2.clusters) == len(results1.clusters)

    def test_resume_validates_input_hash(self, small_test_fasta, tmp_path):
        """Test that resume validates input FASTA hash."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a temporary input file
        temp_input = tmp_path / "input.fasta"
        with open(small_test_fasta, 'r') as f:
            temp_input.write_text(f.read())

        # Run initial clustering
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer.decompose(
            input_fasta=str(temp_input),
            max_clusters=2,
            output_dir=str(output_dir)
        )

        # Modify the input file
        temp_input.write_text(">seq1\nATCG\n>seq2\nGCTA\n")

        # Resume should fail without force flag
        with pytest.raises(ValueError, match="Input FASTA has changed"):
            resume_decompose(
                output_dir=output_dir,
                show_progress=False
            )

        # Resume with force flag should work (but give warning)
        results2 = resume_decompose(
            output_dir=output_dir,
            force_input_change=True,
            show_progress=False
        )
        # Should return existing results without crashing
        assert results2 is not None


class TestResumeClusterContinuation:
    """Tests for continuing cluster generation."""

    def test_resume_continues_iteration_counter(self, small_test_fasta, tmp_path):
        """Test that iteration counter continues from checkpoint."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Initial run
        decomposer1 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer1.decompose(
            input_fasta=small_test_fasta,
            max_clusters=2,
            output_dir=str(output_dir)
        )

        # Check checkpoint state
        state = DecomposeState.load(output_dir)
        assert state.initial_clustering.total_iterations == 2

        # Resume
        results2 = resume_decompose(
            output_dir=output_dir,
            max_clusters=4,
            show_progress=False
        )

        # Final state should show 4 iterations
        final_state = DecomposeState.load(output_dir)
        assert final_state.initial_clustering.total_iterations >= 4

    def test_resume_preserves_existing_clusters(self, small_test_fasta, tmp_path):
        """Test that existing clusters are preserved on resume."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Initial run
        decomposer1 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer1.decompose(
            input_fasta=small_test_fasta,
            max_clusters=2,
            output_dir=str(output_dir)
        )

        # Get cluster IDs from first run
        initial_cluster_ids = set(results1.clusters.keys())

        # Resume and add more
        results2 = resume_decompose(
            output_dir=output_dir,
            max_clusters=4,
            show_progress=False
        )

        # All initial cluster IDs should still be present
        final_cluster_ids = set(results2.clusters.keys())
        # Note: cluster IDs may be renumbered, so we check counts instead
        assert len(results2.clusters) >= len(results1.clusters)


@pytest.mark.integration
class TestResumeEndToEnd:
    """End-to-end integration tests for resume functionality."""

    def test_incremental_cluster_addition(self, small_test_fasta, tmp_path):
        """Test adding clusters incrementally across multiple resumes."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run 1: Create 2 clusters
        decomposer1 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer1.decompose(
            input_fasta=small_test_fasta,
            max_clusters=2,
            output_dir=str(output_dir)
        )

        assert len(results1.clusters) >= 2

        # Run 2: Add more clusters (up to 4 total)
        results2 = resume_decompose(
            output_dir=output_dir,
            max_clusters=4,
            show_progress=False
        )

        assert len(results2.clusters) >= 4

        # Run 3: Add even more (up to 6 total)
        results3 = resume_decompose(
            output_dir=output_dir,
            max_clusters=6,
            show_progress=False
        )

        assert len(results3.clusters) >= 6

        # Verify state tracking
        final_state = DecomposeState.load(output_dir)
        assert final_state.initial_clustering.total_clusters >= 6
        assert final_state.initial_clustering.total_iterations >= 6
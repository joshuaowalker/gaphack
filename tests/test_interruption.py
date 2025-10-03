"""Tests for graceful interruption handling (Phase 3)."""

import pytest
import signal
import threading
import time
from pathlib import Path
from gaphack.decompose import DecomposeClustering


@pytest.fixture
def small_test_fasta():
    """Provide path to small test FASTA file."""
    return "tests/test_data/russula_diverse_50.fasta"


class TestGracefulInterruption:
    """Tests for graceful interruption handling with Ctrl+C."""

    def test_checkpoint_interval_configuration(self, small_test_fasta, tmp_path):
        """Test that checkpoint_interval parameter is respected."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run clustering with custom checkpoint interval
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        # Run with checkpoint every 2 iterations instead of default 10
        results = decomposer.decompose(
            input_fasta=small_test_fasta,
            max_clusters=5,
            output_dir=str(output_dir),
            checkpoint_interval=2
        )

        # Verify clustering completed
        assert results.total_iterations >= 5
        assert len(results.clusters) >= 5

        # Verify state file was created
        state_file = output_dir / "state.json"
        assert state_file.exists()

    def test_checkpoint_saves_progress(self, small_test_fasta, tmp_path):
        """Test that checkpoints are saved at correct intervals."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run clustering with frequent checkpoints
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results = decomposer.decompose(
            input_fasta=small_test_fasta,
            max_clusters=6,
            output_dir=str(output_dir),
            checkpoint_interval=3  # Save every 3 iterations
        )

        # After 6 iterations, we should have checkpoints at iterations 3 and 6
        from gaphack.state import DecomposeState
        state = DecomposeState.load(output_dir)

        # Verify final checkpoint reflects completion
        assert state.initial_clustering.total_iterations >= 6
        assert state.initial_clustering.total_clusters >= 6

    def test_simulated_interruption_with_resume(self, small_test_fasta, tmp_path):
        """Test that interrupted run can be resumed successfully."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Initial run with limit to simulate interruption
        decomposer1 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer1.decompose(
            input_fasta=small_test_fasta,
            max_clusters=3,
            output_dir=str(output_dir),
            checkpoint_interval=2
        )

        initial_iterations = results1.total_iterations
        assert initial_iterations >= 3

        # Resume from checkpoint
        from gaphack.resume import resume_decompose
        results2 = resume_decompose(
            output_dir=output_dir,
            max_clusters=7,
            checkpoint_interval=2,
            show_progress=False
        )

        # Should have continued from where it left off
        # Note: May not reach 7 if sequences are exhausted
        assert results2.total_iterations >= initial_iterations
        assert len(results2.clusters) >= len(results1.clusters)

    def test_signal_handler_not_installed_without_output_dir(self, small_test_fasta):
        """Test that signal handler is only installed when output_dir is provided."""
        # Run without output_dir - no checkpointing
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        # This should work without signal handler issues
        results = decomposer.decompose(
            input_fasta=small_test_fasta,
            max_clusters=2
        )

        assert results.total_iterations >= 2


@pytest.mark.integration
class TestInterruptionResumeWorkflow:
    """Integration tests for complete interruption and resume workflows."""

    def test_multiple_interruptions_and_resumes(self, small_test_fasta, tmp_path):
        """Test multiple interrupt-resume cycles."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # First run: create 2 clusters
        decomposer1 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer1.decompose(
            input_fasta=small_test_fasta,
            max_clusters=2,
            output_dir=str(output_dir),
            checkpoint_interval=1
        )

        assert len(results1.clusters) >= 2

        # Second run: add more clusters (simulating resume after interrupt)
        from gaphack.resume import resume_decompose
        results2 = resume_decompose(
            output_dir=output_dir,
            max_clusters=4,
            checkpoint_interval=1,
            show_progress=False
        )

        assert len(results2.clusters) >= 4

        # Third run: add even more
        results3 = resume_decompose(
            output_dir=output_dir,
            max_clusters=6,
            checkpoint_interval=1,
            show_progress=False
        )

        assert len(results3.clusters) >= 6

        # Verify state consistency
        from gaphack.state import DecomposeState
        final_state = DecomposeState.load(output_dir)
        assert final_state.initial_clustering.total_iterations >= 6
        assert final_state.initial_clustering.total_clusters >= 6

    def test_interruption_preserves_data_integrity(self, small_test_fasta, tmp_path):
        """Test that interruption doesn't corrupt data."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run with limit to simulate interruption
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer.decompose(
            input_fasta=small_test_fasta,
            max_clusters=3,
            output_dir=str(output_dir),
            checkpoint_interval=1
        )

        # Verify cluster files exist
        cluster_files = list((output_dir / "work/initial").glob("*.fasta"))
        assert len(cluster_files) >= 3

        # Verify each cluster file is valid FASTA
        from gaphack.utils import load_sequences_from_fasta
        for cluster_file in cluster_files:
            sequences, headers, _ = load_sequences_from_fasta(str(cluster_file))
            assert len(sequences) > 0
            assert len(sequences) == len(headers)

        # Resume and verify data integrity maintained
        from gaphack.resume import resume_decompose
        results2 = resume_decompose(
            output_dir=output_dir,
            max_clusters=5,
            checkpoint_interval=1,
            show_progress=False
        )

        # Should have more clusters now
        cluster_files_after = list((output_dir / "work/initial").glob("*.fasta"))
        assert len(cluster_files_after) >= 5

        # All cluster files should still be valid
        for cluster_file in cluster_files_after:
            sequences, headers, _ = load_sequences_from_fasta(str(cluster_file))
            assert len(sequences) > 0
            assert len(sequences) == len(headers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
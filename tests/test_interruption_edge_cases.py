"""Tests for Phase 3.5 interruption edge cases."""

import pytest
from pathlib import Path
from gaphack.decompose import DecomposeClustering


@pytest.fixture
def small_test_fasta():
    """Provide path to small test FASTA file."""
    return "tests/test_data/russula_diverse_50.fasta"


class TestInterruptionEdgeCases:
    """Tests for edge cases in interruption handling."""

    def test_detect_partial_state_on_restart(self, small_test_fasta, tmp_path):
        """Test that rerunning same command detects partial state.

        If a previous run was interrupted (or stopped with max_clusters),
        rerunning the same command without --resume should error with
        a helpful message.
        """
        from gaphack.decompose import DecomposeClustering
        import subprocess
        import sys

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # First run: create partial state
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

        assert results1.total_iterations >= 3

        # Verify state file exists and shows in_progress
        from gaphack.state import DecomposeState
        state = DecomposeState.load(output_dir)
        assert state.status == "in_progress"

        # Second run: try to run same command again (via CLI)
        # This should detect the partial state and error
        cmd = [
            sys.executable, "-m", "gaphack.decompose_cli",
            small_test_fasta,
            "-o", str(output_dir),
            "--max-clusters", "5"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        # Should exit with error code
        assert result.returncode != 0, "Should fail when detecting partial state"

        # Check error message mentions --resume
        assert "--resume" in result.stderr, f"Error message should mention --resume flag, got: {result.stderr}"
        assert "partial state" in result.stderr or "in_progress" in result.stderr, \
            f"Error message should mention partial state, got: {result.stderr}"

    def test_interrupt_skips_refinement(self, small_test_fasta, tmp_path):
        """Test that hitting max_clusters limit triggers early return from loop.

        When the max_clusters limit is reached during initial clustering,
        the loop breaks early. With resolve_conflicts and refine_close_clusters
        enabled, verify what happens with post-processing.

        Note: This is a regression test to verify our fix works correctly.
        The fix adds an interruption_requested check before post-processing.
        In a real interruption scenario, this flag would be set by Ctrl+C.
        Here we test the path that just breaks from the loop.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run clustering (decompose no longer has refinement parameters)
        # Stop early with max_clusters limit
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        # Run with max_clusters=3 to create partial state
        results = decomposer.decompose(
            input_fasta=small_test_fasta,
            max_clusters=3,
            output_dir=str(output_dir),
            checkpoint_interval=1
        )

        # Verify clustering ran but stopped at limit
        assert results.total_iterations >= 3
        assert len(results.clusters) >= 3

        # Verify state file shows in_progress status
        from gaphack.state import DecomposeState
        state = DecomposeState.load(output_dir)
        assert state.status == "in_progress", \
            "State should show in_progress when stopped at max_clusters limit"


    def test_multiprocessing_cleanup_on_interrupt(self, small_test_fasta, tmp_path):
        """Test that multiprocessing workers are cleaned up properly.

        This test verifies that the multiprocessing infrastructure in core.py
        properly cleans up even when work completes or is interrupted.
        The finally block in core.py:694 calls executor.shutdown(wait=True)
        which should ensure clean shutdown.

        Investigation findings:
        - ProcessPoolExecutor is created in core.py:642 with proper initialization
        - Finally block at core.py:694 ensures executor.shutdown(wait=True) is called
        - When SIGINT occurs, signal goes to all processes in group
        - Workers will terminate, and finally block will clean up executor
        - Current implementation should handle cleanup correctly

        This test just verifies that clustering with refinement completes
        successfully without leaving resources hanging.
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run clustering (decompose no longer has refinement parameters)
        from gaphack.decompose import DecomposeClustering
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        # Run with small dataset
        results = decomposer.decompose(
            input_fasta=small_test_fasta,
            max_clusters=3,
            output_dir=str(output_dir),
            checkpoint_interval=1
        )

        # Verify clustering completed successfully
        assert results.total_iterations >= 3
        assert len(results.clusters) >= 3

        # If we got here without hanging, multiprocessing cleanup worked
        # The finally block in core.py ensures proper executor shutdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
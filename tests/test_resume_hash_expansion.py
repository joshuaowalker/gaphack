"""Tests for hash ID expansion during resume (Phase 3.6 bugfix)."""

import pytest
from pathlib import Path
from gaphack.decompose import DecomposeClustering, resume_decompose


@pytest.fixture
def small_test_fasta():
    """Provide path to small test FASTA file."""
    return "tests/test_data/russula_diverse_50.fasta"


class TestResumeHashExpansion:
    """Tests for correct hash ID to original header expansion during resume."""

    def test_resume_expands_hash_ids_to_original_headers(self, small_test_fasta, tmp_path):
        """Test that resume properly expands hash IDs back to original headers.

        Bug: When resuming from completed initial clustering, the results
        contained hash IDs (seq_XXXXX) instead of original FASTA headers,
        causing warnings in CLI output.

        Fix: Added expansion step in resume_decompose() when returning results
        from completed clustering (the "no action needed" path).
        """
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # First run: complete initial clustering (exhaustive)
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer.decompose(
            input_fasta=small_test_fasta,
            output_dir=str(output_dir),
            checkpoint_interval=1
        )

        # Verify initial run returns expanded headers (not hash IDs)
        # Original headers should contain "Russula" or similar
        for header in results1.unassigned[:5]:  # Check first 5
            assert not header.startswith("seq_"), \
                f"Initial run should return original headers, not hash IDs: {header}"

        # Second run: resume with no action needed (clustering already complete)
        results2 = resume_decompose(
            output_dir=output_dir,
            show_progress=False
        )

        # Key assertion: resumed results should ALSO have expanded headers
        for header in results2.unassigned[:5]:  # Check first 5
            assert not header.startswith("seq_"), \
                f"Resume should expand hash IDs to original headers, but got: {header}"

        # Verify clusters are also expanded
        for cluster_id, members in results2.clusters.items():
            for header in members[:3]:  # Check first 3 of each cluster
                assert not header.startswith("seq_"), \
                    f"Cluster {cluster_id} should have original headers, not hash IDs: {header}"

    def test_resume_without_duplicates_still_works(self, tmp_path):
        """Test that resume works correctly when input has no duplicates.

        When there are no duplicates, hash_to_headers maps hash_id -> [original_header]
        (single element list), and expansion should still work correctly.
        """
        from gaphack.utils import load_sequences_from_fasta

        # Create a small FASTA with no duplicates
        test_fasta = tmp_path / "input.fasta"
        test_fasta.write_text(
            ">seq1\nACGTACGTACGT\n"
            ">seq2\nTGCATGCATGCA\n"
            ">seq3\nGGGGAAAACCCC\n"
            ">seq4\nTTTTCCCCAAAA\n"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Run decompose to completion
        decomposer = DecomposeClustering(
            min_split=0.001,
            max_lump=0.01,
            target_percentile=95,
            show_progress=False
        )

        results1 = decomposer.decompose(
            input_fasta=str(test_fasta),
            output_dir=str(output_dir)
        )

        # Resume (no action needed)
        results2 = resume_decompose(
            output_dir=output_dir,
            show_progress=False
        )

        # Verify headers are correct (original headers, not hash IDs)
        all_headers = set()
        for cluster_members in results2.clusters.values():
            all_headers.update(cluster_members)
        all_headers.update(results2.unassigned)

        # Should have original headers
        expected_headers = {"seq1", "seq2", "seq3", "seq4"}
        assert all_headers == expected_headers, \
            f"Expected {expected_headers}, got {all_headers}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""Test script for separated conflict resolution and close cluster refinement."""

import tempfile
import logging
from pathlib import Path

from gaphack.decompose import DecomposeClustering


def create_test_fasta(sequences, filename):
    """Create a test FASTA file with given sequences."""
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")


def test_separated_functionality():
    """Test the separated conflict resolution and close cluster refinement."""
    print("=== Testing Separated Conflict Resolution and Close Cluster Refinement ===\n")

    # Create test sequences
    test_sequences = [
        "ATGCGATCGATCGATCG",    # seq_0
        "ATGCGATCGATCGATCC",    # seq_1 - similar to seq_0
        "TTTTTTTTTTTTTTTTTT",   # seq_2 - very different
        "TTTTTTTTTTTTTTTTCC",   # seq_3 - similar to seq_2
        "GGGGGGGGGGGGGGGGGG",   # seq_4 - very different
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create input FASTA
        input_fasta = tmpdir / "input.fasta"
        create_test_fasta(test_sequences, input_fasta)

        # Create targets FASTA
        targets_fasta = tmpdir / "targets.fasta"
        create_test_fasta([test_sequences[0], test_sequences[2], test_sequences[4]], targets_fasta)

        print("Test 1: Pure conflict resolution (minimal scope)")
        decomposer_conflicts_only = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            allow_overlaps=True,
            resolve_conflicts=True,   # Enable conflict resolution
            refine_close_clusters=False,  # Disable close cluster refinement
            close_cluster_threshold=0.0,  # No expansion
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        results_conflicts = decomposer_conflicts_only.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            strategy="supervised"
        )

        print(f"  Clusters: {len(results_conflicts.clusters)}")
        print(f"  Conflicts: {len(results_conflicts.conflicts)}")
        print(f"  Final verification passed: {results_conflicts.verification_results['final']['mece_property']}")
        print()

        print("Test 2: Close cluster refinement with expansion threshold")
        decomposer_refinement = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            allow_overlaps=True,
            resolve_conflicts=False,  # Disable conflict resolution
            refine_close_clusters=True,  # Enable close cluster refinement
            close_cluster_threshold=0.03,  # 3% distance threshold for expansion
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        results_refinement = decomposer_refinement.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            strategy="supervised"
        )

        print(f"  Clusters: {len(results_refinement.clusters)}")
        print(f"  Conflicts: {len(results_refinement.conflicts)}")
        print(f"  Final verification passed: {results_refinement.verification_results['final']['mece_property']}")
        print()

        print("Test 3: Both conflict resolution and refinement")
        decomposer_both = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            allow_overlaps=True,
            resolve_conflicts=True,   # Enable conflict resolution
            refine_close_clusters=True,  # Enable close cluster refinement
            close_cluster_threshold=0.025,  # 2.5% distance threshold
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        results_both = decomposer_both.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            strategy="supervised"
        )

        print(f"  Clusters: {len(results_both.clusters)}")
        print(f"  Conflicts: {len(results_both.conflicts)}")
        print(f"  Final verification passed: {results_both.verification_results['final']['mece_property']}")
        print()

        print("Test 4: CLI-style usage demonstration")
        print("  # Pure conflict resolution (minimal scope):")
        print("  gaphack-decompose input.fasta --targets targets.fasta --resolve-conflicts -o results")
        print()
        print("  # Close cluster refinement with 2% expansion threshold:")
        print("  gaphack-decompose input.fasta --targets targets.fasta --refine-close-clusters 0.02 -o results")
        print()
        print("  # Both correctness and quality improvement:")
        print("  gaphack-decompose input.fasta --targets targets.fasta --resolve-conflicts --refine-close-clusters 0.025 -o results")
        print()

        print("✓ All separated functionality tests completed successfully!")
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_separated_functionality()
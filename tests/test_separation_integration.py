#!/usr/bin/env python3
"""Test script to demonstrate common usage patterns with separated decompose and refine CLIs.

NOTE: This test demonstrates the three common usage patterns with the new
modular gaphack-decompose and gaphack-refine architecture.
"""

import tempfile
import logging
import pytest
from pathlib import Path
from Bio import SeqIO

from gaphack.decompose import DecomposeClustering
from test_phase4_integration import CLIRunner


def create_test_fasta(sequences, filename):
    """Create a test FASTA file with given sequences."""
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")


def test_usage_patterns():
    """Test the three common usage patterns for gaphack-decompose and gaphack-refine."""
    print("=== Testing Common Usage Patterns ===\n")

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

        print("=" * 70)
        print("Pattern 1: Decompose only (no refinement)")
        print("=" * 70)
        print("  CLI command:")
        print("    gaphack-decompose input.fasta --targets targets.fasta -o results/")
        print()

        output_dir_1 = tmpdir / "pattern1_decompose_only"
        decomposer_1 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        decomposer_1.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            output_dir=str(output_dir_1)
        )

        clusters_dir_1 = output_dir_1 / "work" / "initial"
        cluster_files_1 = list(clusters_dir_1.glob("cluster_*.fasta"))
        print(f"  Result: {len(cluster_files_1)} clusters")
        print(f"  Use case: Quick clustering, conflicts acceptable for downstream processing")
        print()

        print("=" * 70)
        print("Pattern 2: Decompose + conflict resolution only")
        print("=" * 70)
        print("  CLI commands:")
        print("    gaphack-decompose input.fasta --targets targets.fasta -o results/")
        print("    gaphack-refine --input-dir results/work/initial/ --output-dir refined/")
        print()

        output_dir_2 = tmpdir / "pattern2_decompose"
        decomposer_2 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        decomposer_2.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            output_dir=str(output_dir_2)
        )

        # Run gaphack-refine (conflict resolution only)
        clusters_dir_2 = output_dir_2 / "work" / "initial"
        refine_output_2 = tmpdir / "pattern2_refined"
        result_2 = CLIRunner.run_gaphack_refine(
            input_dir=clusters_dir_2,
            output_dir=refine_output_2,
            no_timestamp=True
        )

        assert result_2['returncode'] == 0, f"Refinement should succeed. stderr: {result_2['stderr']}"
        cluster_files_2 = list(refine_output_2.glob("cluster_*.fasta"))
        print(f"  Result: {len(cluster_files_2)} MECE clusters (no conflicts)")
        print(f"  Use case: Ensure mutually exclusive clustering for downstream analysis")
        print()

        print("=" * 70)
        print("Pattern 3: Decompose + full refinement (conflicts + close clusters)")
        print("=" * 70)
        print("  CLI commands:")
        print("    gaphack-decompose input.fasta --targets targets.fasta -o results/")
        print("    gaphack-refine --input-dir results/work/initial/ \\")
        print("                   --output-dir refined/ \\")
        print("                   --refine-close-clusters 0.025")
        print()

        output_dir_3 = tmpdir / "pattern3_decompose"
        decomposer_3 = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        decomposer_3.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            output_dir=str(output_dir_3)
        )

        # Run gaphack-refine with close cluster refinement
        clusters_dir_3 = output_dir_3 / "work" / "initial"
        refine_output_3 = tmpdir / "pattern3_refined"
        result_3 = CLIRunner.run_gaphack_refine(
            input_dir=clusters_dir_3,
            output_dir=refine_output_3,
            refine_close_clusters=0.025,  # 2.5% distance threshold
            no_timestamp=True
        )

        assert result_3['returncode'] == 0, f"Refinement should succeed. stderr: {result_3['stderr']}"
        cluster_files_3 = list(refine_output_3.glob("cluster_*.fasta"))
        print(f"  Result: {len(cluster_files_3)} optimized MECE clusters")
        print(f"  Use case: Maximum quality - resolve conflicts AND optimize barcode gaps")
        print()

        # Verify MECE property for patterns 2 and 3
        for pattern_num, refine_output in [(2, refine_output_2), (3, refine_output_3)]:
            all_sequences = set()
            cluster_files = list(refine_output.glob("cluster_*.fasta"))
            for cluster_file in cluster_files:
                cluster_sequences = {record.id for record in SeqIO.parse(cluster_file, "fasta")}
                overlap = all_sequences & cluster_sequences
                assert len(overlap) == 0, f"Pattern {pattern_num}: MECE violation - {overlap}"
                all_sequences.update(cluster_sequences)

        print("=" * 70)
        print("✓ All usage patterns completed successfully!")
        print("=" * 70)


# Separation integration test converted to pytest format
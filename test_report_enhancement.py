#!/usr/bin/env python3
"""Test script to verify enhanced report generation."""

import tempfile
import logging
from pathlib import Path

from gaphack.decompose import DecomposeClustering

def create_test_fasta(sequences, filename):
    """Create a test FASTA file with given sequences."""
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")

def test_enhanced_report():
    """Test enhanced report generation."""
    print("=== Testing Enhanced Report Generation ===\n")

    # Create test sequences that will generate some interesting behavior
    test_sequences = [
        "ATGCGATCGATCGATCGATGC",    # seq_0
        "ATGCGATCGATCGATCGATCC",    # seq_1 - similar to seq_0
        "ATGCGATCGATCGATCGCCCC",    # seq_2 - somewhat similar to seq_0/1
        "TTTTTTTTTTTTTTTTTTTT",     # seq_3 - very different
        "TTTTTTTTTTTTTTTTTTCC",     # seq_4 - similar to seq_3
        "GGGGGGGGGGGGGGGGGGGG",     # seq_5 - very different
        "GGGGGGGGGGGGGGGGGGCC",     # seq_6 - similar to seq_5
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create input FASTA
        input_fasta = tmpdir / "input.fasta"
        create_test_fasta(test_sequences, input_fasta)

        # Create targets FASTA (subset that should create some conflicts)
        targets_fasta = tmpdir / "targets.fasta"
        create_test_fasta([test_sequences[0], test_sequences[3], test_sequences[5]], targets_fasta)

        print(f"Test dataset: {len(test_sequences)} sequences, 3 targets")
        print("Running with both conflict resolution and close cluster refinement...")

        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            allow_overlaps=True,
            resolve_conflicts=True,
            refine_close_clusters=True,
            close_cluster_threshold=0.03,  # 3% distance threshold
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        results = decomposer.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            strategy="supervised"
        )

        # Save report to check formatting
        from gaphack.decompose_cli import save_decompose_results
        import sys
        import datetime

        # Add metadata for reporting
        results.command_line = ' '.join(sys.argv)
        results.start_time = datetime.datetime.now().isoformat()

        output_base = tmpdir / "test_results"
        save_decompose_results(results, str(output_base), str(input_fasta))

        # Read and display the report
        report_file = f"{output_base}.decompose_report.txt"
        print(f"\n=== Generated Report ===")
        with open(report_file, 'r') as f:
            report_content = f.read()
            print(report_content)

        print("✓ Enhanced report generation test completed!")
        return True

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise
    test_enhanced_report()
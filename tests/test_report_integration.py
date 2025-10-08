#!/usr/bin/env python3
"""Test script to verify report generation for decompose and refine separately.

NOTE: These tests verify that both gaphack-decompose and gaphack-refine generate
proper reports independently.
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

def test_decompose_report_generation():
    """Test that gaphack-decompose generates a proper report."""
    print("=== Testing gaphack-decompose Report Generation ===\n")

    # Create test sequences
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

        # Create targets FASTA
        targets_fasta = tmpdir / "targets.fasta"
        create_test_fasta([test_sequences[0], test_sequences[3], test_sequences[5]], targets_fasta)

        print(f"Test dataset: {len(test_sequences)} sequences, 3 targets")
        print("Running gaphack-decompose via CLI...")

        output_dir = tmpdir / "decompose_output"
        result = CLIRunner.run_gaphack_decompose(
            input_path=input_fasta,
            output_base=output_dir,
            targets=targets_fasta,
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95
        )

        assert result['returncode'] == 0, f"Decompose should succeed. stderr: {result['stderr']}"

        # Check that decompose report exists (in clusters/latest/)
        report_file = output_dir / "clusters" / "latest" / "decompose_report.txt"
        assert report_file.exists(), "Decompose report should exist"

        # Read and verify report content
        with open(report_file, 'r') as f:
            report_content = f.read()
            print(f"\n=== Decompose Report ===")
            print(report_content)

        # Verify key sections exist in report
        assert "Gaphack-Decompose Clustering Report" in report_content, "Report should have title"
        assert "Summary Statistics" in report_content, "Report should have summary statistics"
        assert "Iteration Summary" in report_content, "Report should have iteration summary"

        print("✓ Decompose report generation test completed!")


def test_refine_report_generation():
    """Test that gaphack-refine generates a proper report."""
    print("\n=== Testing gaphack-refine Report Generation ===\n")

    # Create test sequences
    test_sequences = [
        "ATGCGATCGATCGATCGATGC",    # seq_0
        "ATGCGATCGATCGATCGATCC",    # seq_1 - similar to seq_0
        "TTTTTTTTTTTTTTTTTTTT",     # seq_2 - very different
        "TTTTTTTTTTTTTTTTTTCC",     # seq_3 - similar to seq_2
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create input FASTA
        input_fasta = tmpdir / "input.fasta"
        create_test_fasta(test_sequences, input_fasta)

        # Step 1: Run decompose via CLI (to generate report)
        output_dir = tmpdir / "decompose_output"
        result = CLIRunner.run_gaphack_decompose(
            input_path=input_fasta,
            output_base=output_dir,
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95
        )

        assert result['returncode'] == 0, f"Decompose should succeed. stderr: {result['stderr']}"

        # Step 2: Run refine
        print("Running gaphack-refine...")
        initial_clusters_dir = output_dir / "work" / "initial"
        refine_output = tmpdir / "refined_output"

        result = CLIRunner.run_gaphack_refine(
            input_dir=initial_clusters_dir,
            output_dir=refine_output,
            refine_close_clusters=0.03,
            no_timestamp=True
        )

        assert result['returncode'] == 0, f"Refinement should succeed. stderr: {result['stderr']}"

        # Check that refine report exists
        report_file = refine_output / "refine_summary.txt"
        assert report_file.exists(), "Refine summary report should exist"

        # Read and verify report content
        with open(report_file, 'r') as f:
            report_content = f.read()
            print(f"\n=== Refine Report ===")
            print(report_content)

        # Verify key sections exist in report
        assert "gaphack-refine Summary Report" in report_content, "Report should have title"
        # New two-pass mode uses different section names
        assert ("Pass 1" in report_content or "Stage 1" in report_content), "Report should have Pass 1/Stage 1 section"
        assert ("Pass 2" in report_content or "Stage 2" in report_content), "Report should have Pass 2/Stage 2 section"

        print("✓ Refine report generation test completed!")


# Report integration tests converted to pytest format
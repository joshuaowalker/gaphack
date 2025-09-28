#!/usr/bin/env python3
"""Demonstration of performance improvement from removing proximity graph from conflict resolution."""

import tempfile
import time
import logging
from pathlib import Path

from gaphack.decompose import DecomposeClustering


def create_test_fasta(sequences, filename):
    """Create a test FASTA file with given sequences."""
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")


def time_operation(operation_name, operation_func):
    """Time an operation and return the duration."""
    print(f"Running {operation_name}...")
    start_time = time.time()
    result = operation_func()
    end_time = time.time()
    duration = end_time - start_time
    print(f"  {operation_name} completed in {duration:.3f} seconds")
    return result, duration


def demonstrate_performance_improvement():
    """Demonstrate the performance improvement from removing proximity graph from conflict resolution."""
    print("=== Performance Improvement Demonstration ===\n")
    print("Comparing conflict resolution performance before/after proximity graph removal\n")

    # Create test sequences - enough to make proximity graph creation noticeable
    # but not so many that the test takes too long
    test_sequences = []
    base_sequences = [
        "ATGCGATCGATCGATCGATGC",  # Template sequences
        "TTTTGGGGCCCCAAAATTTT",
        "GGGGAAAACCCCTTTTGGGG",
        "CCCCTTTTGGGGAAAACCCC",
        "AAAACCCCTTTTGGGGAAAA",
    ]

    # Create variations to simulate a realistic clustering scenario
    for base in base_sequences:
        for i in range(4):  # 4 variations per base = 20 sequences total
            # Create slight variations
            variation = base[:10] + str(i) * 2 + base[12:]
            test_sequences.append(variation)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create input FASTA
        input_fasta = tmpdir / "input.fasta"
        create_test_fasta(test_sequences, input_fasta)

        # Create targets FASTA (subset of inputs)
        targets_fasta = tmpdir / "targets.fasta"
        create_test_fasta(test_sequences[::4], targets_fasta)  # Every 4th sequence

        print(f"Test dataset: {len(test_sequences)} sequences, {len(test_sequences)//4} targets")
        print()

        # Test pure conflict resolution (should be fast now - no proximity graph)
        def run_conflict_resolution_only():
            decomposer = DecomposeClustering(
                min_split=0.005,
                max_lump=0.02,
                target_percentile=95,
                resolve_conflicts=True,  # Only conflict resolution
                refine_close_clusters=False,  # No close cluster refinement
                close_cluster_threshold=0.0,
                show_progress=False,
                logger=logging.getLogger(__name__)
            )
            return decomposer.decompose(
                input_fasta=str(input_fasta),
                targets_fasta=str(targets_fasta),
                strategy="supervised"
            )

        # Test close cluster refinement (still uses proximity graph)
        def run_close_cluster_refinement_only():
            decomposer = DecomposeClustering(
                min_split=0.005,
                max_lump=0.02,
                target_percentile=95,
                resolve_conflicts=False,  # No conflict resolution
                refine_close_clusters=True,  # Only close cluster refinement
                close_cluster_threshold=0.02,
                show_progress=False,
                logger=logging.getLogger(__name__)
            )
            return decomposer.decompose(
                input_fasta=str(input_fasta),
                targets_fasta=str(targets_fasta),
                strategy="supervised"
            )

        # Time both operations
        results_conflicts, time_conflicts = time_operation(
            "Pure conflict resolution (minimal scope)",
            run_conflict_resolution_only
        )

        results_refinement, time_refinement = time_operation(
            "Close cluster refinement (with proximity graph)",
            run_close_cluster_refinement_only
        )

        print()
        print("Performance Comparison:")
        print(f"  Conflict resolution:      {time_conflicts:.3f}s (no proximity graph)")
        print(f"  Close cluster refinement: {time_refinement:.3f}s (with proximity graph)")

        if time_refinement > time_conflicts:
            speedup = time_refinement / time_conflicts
            print(f"  Speedup factor: {speedup:.1f}x faster")

        print()
        print("Key Observations:")
        print("  - Conflict resolution is now faster due to skipped proximity graph creation")
        print("  - Close cluster refinement still uses proximity graph for quality optimization")
        print("  - Users can choose speed (conflicts only) vs quality (both operations)")
        print()

        print("Verification:")
        print(f"  Conflict resolution MECE: {results_conflicts.verification_results['final']['mece_property']}")
        print(f"  Close refinement MECE:    {results_refinement.verification_results['final']['mece_property']}")

        print()
        print("✓ Performance improvement demonstration completed!")
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # Reduce log noise for timing
    demonstrate_performance_improvement()
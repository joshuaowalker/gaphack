#!/usr/bin/env python3
"""Test script for principled reclustering integration with gaphack-refine.

NOTE: These tests now use the gaphack-refine CLI to test refinement functionality
that has been separated from gaphack-decompose.
"""

import tempfile
import logging
import pytest
from pathlib import Path
from Bio import SeqIO

from gaphack.decompose import DecomposeClustering
from gaphack.cluster_refinement import verify_no_conflicts
from test_phase4_integration import CLIRunner


def create_test_fasta(sequences, filename):
    """Create a test FASTA file with given sequences."""
    with open(filename, 'w') as f:
        for i, seq in enumerate(sequences):
            f.write(f">seq_{i}\n{seq}\n")


def test_conflict_resolution_integration():
    """Test the basic conflict resolution integration using gaphack-refine CLI."""
    print("Testing gaphack-refine conflict resolution integration...")

    # Create test sequences that will likely create conflicts
    test_sequences = [
        "ATGCGATCGATCGATCG",    # seq_0
        "ATGCGATCGATCGATCC",    # seq_1 - similar to seq_0
        "ATGCGATCGATCGATCA",    # seq_2 - similar to seq_0 and seq_1
        "TTTTTTTTTTTTTTTTTT",   # seq_3 - very different
        "TTTTTTTTTTTTTTTTCC",   # seq_4 - similar to seq_3
        "GGGGGGGGGGGGGGGGGG",   # seq_5 - very different from all
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create input FASTA
        input_fasta = tmpdir / "input.fasta"
        create_test_fasta(test_sequences, input_fasta)

        # Create targets FASTA (use first and third sequences as targets)
        targets_fasta = tmpdir / "targets.fasta"
        create_test_fasta([test_sequences[0], test_sequences[3]], targets_fasta)

        # Step 1: Run gaphack-decompose
        output_dir = tmpdir / "decompose_output"
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,  # Low threshold to encourage conflicts
            target_percentile=95,
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        print(f"Running decomposition on {len(test_sequences)} sequences...")
        decomposer.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            output_dir=str(output_dir)
        )

        # Step 2: Run gaphack-refine for conflict resolution
        initial_clusters_dir = output_dir / "work" / "initial"
        refine_output = tmpdir / "refined_output"
        result = CLIRunner.run_gaphack_refine(
            input_dir=initial_clusters_dir,
            output_dir=refine_output,
            no_timestamp=True
        )

        assert result['returncode'] == 0, f"Refinement should succeed. stderr: {result['stderr']}"

        # Analyze results
        refined_files = list(refine_output.glob("cluster_*.fasta"))
        print(f"Results:")
        print(f"  Total clusters: {len(refined_files)}")

        # Verify MECE property
        all_assigned_sequences = set()
        for cluster_file in refined_files:
            cluster_sequences = {record.id for record in SeqIO.parse(cluster_file, "fasta")}
            overlap = all_assigned_sequences & cluster_sequences
            if overlap:
                print(f"  ❌ MECE violation: {overlap} appears in multiple clusters")
                assert False, f"MECE violation: {overlap}"
            all_assigned_sequences.update(cluster_sequences)

        print(f"  ✓ MECE property verified - no sequence appears in multiple clusters")
        print(f"  ✓ All conflicts resolved!")


def test_conflict_resolution_algorithm_directly():
    """Test the MECE verification algorithm directly."""
    print("\nTesting MECE verification algorithm directly...")

    # Create clusters with known conflicts (seq_1 in both clusters)
    all_clusters = {
        "cluster_1": ["seq_0", "seq_1"],
        "cluster_2": ["seq_1", "seq_2"],
        "cluster_3": ["seq_3"]
    }

    print(f"Input clusters: {all_clusters}")

    # Verify conflicts are detected
    verification = verify_no_conflicts(all_clusters, context="test")
    print(f"Conflicts detected: {verification['conflicts']}")

    assert not verification['no_conflicts'], "Should detect conflicts"
    assert verification['conflict_count'] == 1, "Should detect 1 conflicted sequence"
    assert "seq_1" in verification['conflicts'], "Should detect seq_1 as conflicted"
    assert len(verification['conflicts']['seq_1']) == 2, "seq_1 should be in 2 clusters"

    print("✓ MECE verification algorithm works correctly")


def test_comprehensive_verification():
    """Test the comprehensive MECE verification system."""
    print("\nTesting comprehensive MECE verification system...")

    from gaphack.cluster_refinement import verify_no_conflicts

    # Test Case 1: MECE clusters (no conflicts)
    print("Test 1: MECE clusters (should pass)")
    mece_clusters = {
        "cluster_1": ["seq_A", "seq_B"],
        "cluster_2": ["seq_C", "seq_D"],
        "cluster_3": ["seq_E"]
    }

    verification_1 = verify_no_conflicts(mece_clusters, context="test_mece")
    assert verification_1['no_conflicts'] == True, "MECE clusters should satisfy MECE property"
    assert verification_1['conflict_count'] == 0, "MECE clusters should have no conflicts"
    print("  ✓ MECE verification passed")

    # Test Case 2: Conflicted clusters
    print("Test 2: Conflicted clusters (should detect conflicts)")
    conflicted_clusters = {
        "cluster_1": ["seq_A", "seq_B", "seq_X"],
        "cluster_2": ["seq_C", "seq_X", "seq_D"],  # seq_X in both clusters
        "cluster_3": ["seq_E", "seq_Y"],
        "cluster_4": ["seq_Y", "seq_F"]  # seq_Y in both clusters
    }

    verification_2 = verify_no_conflicts(conflicted_clusters, context="test_conflicts")
    assert verification_2['no_conflicts'] == False, "Conflicted clusters should violate MECE property"
    assert verification_2['conflict_count'] == 2, "Should detect 2 conflicted sequences"
    assert "seq_X" in verification_2['conflicts'], "Should detect seq_X conflict"
    assert "seq_Y" in verification_2['conflicts'], "Should detect seq_Y conflict"
    print("  ✓ Conflict detection passed")

    # Test Case 3: Multi-cluster conflicts
    print("Test 3: Multi-cluster conflicts (sequence in 3+ clusters)")
    multi_conflict_clusters = {
        "cluster_1": ["seq_A", "seq_MULTI"],
        "cluster_2": ["seq_B", "seq_MULTI"],
        "cluster_3": ["seq_C", "seq_MULTI"],
        "cluster_4": ["seq_D"]
    }

    verification_3 = verify_no_conflicts(multi_conflict_clusters, context="test_multi_conflicts")
    assert verification_3['no_conflicts'] == False, "Multi-cluster conflicts should violate MECE property"
    assert verification_3['conflict_count'] == 1, "Should detect 1 conflicted sequence"
    assert "seq_MULTI" in verification_3['conflicts'], "Should detect seq_MULTI conflict"
    assert len(verification_3['conflicts']['seq_MULTI']) == 3, "seq_MULTI should be in 3 clusters"
    print("  ✓ Multi-cluster conflict detection passed")

    # Test Case 4: Resolution tracking
    print("Test 4: Resolution tracking (compare original vs resolved)")
    original_conflicts = {
        "seq_X": ["cluster_1", "cluster_2"],
        "seq_Y": ["cluster_3", "cluster_4"],
        "seq_Z": ["cluster_5", "cluster_6"]
    }

    # Simulate partial resolution (seq_X and seq_Y resolved, seq_Z still conflicted, new conflict introduced)
    partially_resolved_clusters = {
        "cluster_1": ["seq_A", "seq_X"],  # seq_X resolved to cluster_1
        "cluster_2": ["seq_B"],
        "cluster_3": ["seq_C", "seq_Y"],  # seq_Y resolved to cluster_3
        "cluster_4": ["seq_D"],
        "cluster_5": ["seq_E", "seq_Z"],  # seq_Z still conflicted
        "cluster_6": ["seq_F", "seq_Z"],
        "cluster_7": ["seq_G", "seq_NEW"], # new conflict introduced
        "cluster_8": ["seq_H", "seq_NEW"]
    }

    verification_4 = verify_no_conflicts(
        partially_resolved_clusters,
        original_conflicts=original_conflicts,
        context="test_resolution_tracking"
    )

    assert len(verification_4['resolved_conflicts']) == 2, "Should detect 2 resolved conflicts"
    assert "seq_X" in verification_4['resolved_conflicts'], "seq_X should be marked as resolved"
    assert "seq_Y" in verification_4['resolved_conflicts'], "seq_Y should be marked as resolved"
    assert len(verification_4['unresolved_conflicts']) == 1, "Should detect 1 unresolved conflict"
    assert "seq_Z" in verification_4['unresolved_conflicts'], "seq_Z should be marked as unresolved"
    assert len(verification_4['new_conflicts']) == 1, "Should detect 1 new conflict"
    assert "seq_NEW" in verification_4['new_conflicts'], "seq_NEW should be marked as new conflict"
    print("  ✓ Resolution tracking passed")

    print("✓ All comprehensive verification tests passed!")
    # Test completed successfully


def test_verification_integration():
    """Test that MECE verification works correctly across decompose and refine."""
    print("\nTesting MECE verification across decompose and refine workflow...")

    # Create test sequences
    test_sequences = [
        "ATGCGATCGATCGATCG",    # seq_0
        "ATGCGATCGATCGATCC",    # seq_1 - similar to seq_0
        "TTTTTTTTTTTTTTTTTT",   # seq_2 - very different
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create input FASTA
        input_fasta = tmpdir / "input.fasta"
        create_test_fasta(test_sequences, input_fasta)

        # Create targets FASTA
        targets_fasta = tmpdir / "targets.fasta"
        create_test_fasta([test_sequences[0], test_sequences[2]], targets_fasta)

        # Test workflow: decompose -> refine
        print("Testing decompose -> refine workflow...")
        output_dir = tmpdir / "decompose_output"
        decomposer = DecomposeClustering(
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            show_progress=False,
            logger=logging.getLogger(__name__)
        )

        decomposer.decompose(
            input_fasta=str(input_fasta),
            targets_fasta=str(targets_fasta),
            output_dir=str(output_dir)
        )

        # Run gaphack-refine
        initial_clusters_dir = output_dir / "work" / "initial"
        refine_output = tmpdir / "refined_output"
        result = CLIRunner.run_gaphack_refine(
            input_dir=initial_clusters_dir,
            output_dir=refine_output,
            no_timestamp=True
        )

        assert result['returncode'] == 0, f"Refinement should succeed. stderr: {result['stderr']}"

        # Verify MECE property in refined output
        refined_files = list(refine_output.glob("cluster_*.fasta"))
        refined_clusters = {}
        for cluster_file in refined_files:
            cluster_id = cluster_file.stem
            refined_clusters[cluster_id] = [record.id for record in SeqIO.parse(cluster_file, "fasta")]

        verification = verify_no_conflicts(refined_clusters, context="refined_output")
        print(f"  Refined output verification: {verification['conflict_count']} conflicts, MECE: {verification['no_conflicts']}")

        assert verification['no_conflicts'], "Refined output should satisfy MECE property"
        assert verification['conflict_count'] == 0, "Refined output should have no conflicts"

        print("✓ MECE verification integration tests passed!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    success = True

    try:
        success &= test_conflict_resolution_integration()
        success &= test_conflict_resolution_algorithm_directly()
        success &= test_comprehensive_verification()
        success &= test_verification_integration()

        if success:
            print("\n🎉 All integration tests passed!")
        else:
            print("\n❌ Some tests failed")

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        success = False

    exit(0 if success else 1)
#!/usr/bin/env python3
"""Demo script showing final verification catching missed conflicts."""

import logging
from gaphack.cluster_refinement import verify_no_conflicts

def demo_final_verification_catching_missed_conflicts():
    """Demonstrate how final verification catches conflicts that initial detection missed."""

    print("=== Demo: Final Verification Catching Missed Conflicts ===\n")

    # Simulate a scenario where initial processing reported no conflicts
    # but final verification discovers conflicts in the actual cluster assignments

    print("Scenario: Initial processing reported 'no conflicts detected'")
    print("But final cluster assignments actually contain conflicts...\n")

    # This represents what the final cluster assignments actually look like
    final_cluster_assignments = {
        "cluster_001": ["seq_A", "seq_B", "seq_CONFLICT"],
        "cluster_002": ["seq_C", "seq_CONFLICT", "seq_D"],  # seq_CONFLICT in both!
        "cluster_003": ["seq_E", "seq_F"],
        "cluster_004": ["seq_G", "seq_MULTI"],
        "cluster_005": ["seq_H", "seq_MULTI"],  # seq_MULTI in multiple clusters
        "cluster_006": ["seq_I", "seq_MULTI"]   # seq_MULTI in 3 clusters total!
    }

    # Simulate that initial processing thought there were no conflicts
    initial_conflicts_detected = {}

    print("Initial conflict detection result: No conflicts")
    print("Final cluster assignments:")
    for cluster_id, sequences in final_cluster_assignments.items():
        print(f"  {cluster_id}: {sequences}")
    print()

    # Now run final verification - this will catch what was missed
    print("Running final comprehensive verification...")
    verification_result = verify_cluster_assignments_mece(
        clusters=final_cluster_assignments,
        original_conflicts=initial_conflicts_detected,
        context="final_comprehensive"
    )

    print(f"\nFinal Verification Results:")
    print(f"  MECE Property Satisfied: {verification_result['mece_property']}")
    print(f"  Conflicts Detected: {verification_result['conflict_count']}")
    print(f"  Critical Failure: {verification_result['critical_failure']}")

    if verification_result['conflicts']:
        print(f"\nConflicts found by final verification:")
        for seq_id, cluster_ids in verification_result['conflicts'].items():
            print(f"  {seq_id} appears in clusters: {cluster_ids}")

    print(f"\nThis demonstrates how final verification serves as a safety net")
    print(f"to catch conflicts that may be missed by initial detection or")
    print(f"introduced during processing steps like classic gapHACk reclustering.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_final_verification_catching_missed_conflicts()
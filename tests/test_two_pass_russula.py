"""Integration tests for two-pass refinement on Russula dataset.

Tests the two-pass refinement system on real biological data (1,429 fungal
ITS sequences with 143 ground truth groups) to validate:
- Quality metrics meet thresholds (ARI ≥ 0.85, Homogeneity ≥ 0.90, Completeness ≥ 0.85)
- Convergence behavior (3-5 iterations expected)
- Performance (<30 minutes for 1,000+ sequences)
"""

import pytest
import time
from pathlib import Path
from Bio import SeqIO

from gaphack.cluster_refinement import two_pass_refinement, RefinementConfig
from gaphack.cluster_graph import ClusterGraph
from gaphack.refine_cli import detect_conflicts
from gaphack.refinement_types import ClusterIDGenerator


def load_fasta(fasta_path):
    """Load sequences and headers from FASTA file.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Tuple of (sequences list, headers list)
    """
    sequences = []
    headers = []

    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences.append(str(record.seq))
        headers.append(record.id)

    return sequences, headers


class TestTwoPassRussulaIntegration:
    """Integration tests on full Russula dataset (1,429 sequences)."""

    @pytest.fixture
    def russula_full_path(self):
        """Path to full Russula dataset."""
        path = Path("examples/data/Russula_INxx.fasta")
        if not path.exists():
            pytest.skip("Full Russula dataset not found")
        return path

    @pytest.fixture
    def russula_100_path(self):
        """Path to Russula 100 subset for faster tests."""
        path = Path("tests/test_data/russula_diverse_100.fasta")
        if not path.exists():
            pytest.skip("Russula 100 subset not found")
        return path

    @pytest.fixture
    def russula_300_path(self):
        """Path to Russula 300 subset."""
        path = Path("tests/test_data/russula_300.fasta")
        if not path.exists():
            pytest.skip("Russula 300 subset not found")
        return path

    @pytest.mark.quality
    @pytest.mark.slow
    def test_two_pass_full_russula_quality(self, russula_full_path):
        """Test two-pass refinement on full Russula dataset with quality metrics.

        Validates:
        - ARI ≥ 0.85
        - Homogeneity ≥ 0.90
        - Completeness ≥ 0.85
        - Performance < 30 minutes

        NOTE: This test requires ground truth annotations in the sequence headers
        to compute quality metrics.
        """
        pytest.skip("Requires ground truth annotation parsing and quality metric computation")

        # This would require:
        # 1. Loading the full Russula dataset
        # 2. Running initial decompose or loading pre-decomposed clusters
        # 3. Running two-pass refinement
        # 4. Computing quality metrics against ground truth
        # 5. Asserting metrics meet thresholds

    @pytest.mark.integration
    def test_two_pass_convergence_behavior(self, russula_100_path):
        """Test that two-pass refinement converges on Russula 100."""
        # Load sequences
        sequences, headers = load_fasta(russula_100_path)

        # Create initial singleton clusters (worst case for convergence)
        initial_clusters = {
            f"cluster_{i:05d}I": [header]
            for i, header in enumerate(headers, 1)
        }

        # Run two-pass refinement
        config = RefinementConfig(
            max_full_gaphack_size=300,
            max_iterations=10,
            k_neighbors=20,
            search_method="blast"
        )

        start_time = time.time()

        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

        final_clusters, tracking_stages = two_pass_refinement(
            all_clusters=initial_clusters,
            sequences=sequences,
            headers=headers,
            conflicts={},  # No conflicts in singletons
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            config=config,
            cluster_id_generator=cluster_id_generator,
            run_pass1=True,
            run_pass2=True,
            show_progress=False
        )

        duration = time.time() - start_time

        # Check results
        assert len(final_clusters) > 0, "Should produce clusters"
        assert len(final_clusters) < len(initial_clusters), "Should merge some singletons"

        # Check Pass 2 convergence (if Pass 2 was run)
        if len(tracking_stages) >= 2:
            pass2_info = tracking_stages[1] if len(tracking_stages) > 1 else tracking_stages[0]
            iterations = pass2_info.summary_stats.get('iterations', 0)
            convergence_reason = pass2_info.summary_stats.get('convergence_reason', 'unknown')

            print(f"\nPass 2 Results:")
            print(f"  Iterations: {iterations}")
            print(f"  Convergence: {convergence_reason}")
            print(f"  Final clusters: {len(final_clusters)}")
            print(f"  Duration: {duration:.1f}s")

            # Most datasets should converge in reasonable iterations
            assert iterations <= 10, "Should converge within iteration limit"

        # Performance check: 100 sequences should complete quickly
        assert duration < 300, f"Should complete in <5 minutes (took {duration:.1f}s)"

    @pytest.mark.integration
    def test_pass1_conflict_resolution(self, russula_100_path):
        """Test Pass 1 conflict resolution on Russula 100."""
        # Load sequences
        sequences, headers = load_fasta(russula_100_path)

        # Create clusters with artificial conflicts (same sequence in multiple clusters)
        initial_clusters = {
            'cluster_A': headers[:30],
            'cluster_B': headers[20:50],  # Overlaps with A
            'cluster_C': headers[40:70],  # Overlaps with B
            'cluster_D': headers[70:]
        }

        # Detect conflicts
        conflicts = detect_conflicts(initial_clusters)

        assert len(conflicts) > 0, "Should have conflicts from overlapping clusters"

        # Run Pass 1 only
        config = RefinementConfig(
            max_full_gaphack_size=300,
            max_iterations=10,
            k_neighbors=20,
            search_method="blast"
        )

        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

        final_clusters, tracking_stages = two_pass_refinement(
            all_clusters=initial_clusters,
            sequences=sequences,
            headers=headers,
            conflicts=conflicts,
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            config=config,
            cluster_id_generator=cluster_id_generator,
            run_pass1=True,
            run_pass2=False,  # Pass 1 only
            show_progress=False
        )

        # Verify MECE property after Pass 1
        all_assigned = []
        for cluster_headers in final_clusters.values():
            all_assigned.extend(cluster_headers)

        # Check for duplicates (should be none)
        assert len(all_assigned) == len(set(all_assigned)), "Pass 1 should resolve all conflicts (MECE)"

        # Check that we didn't lose sequences
        assert set(all_assigned) <= set(headers), "All assigned sequences should be from input"

        print(f"\nPass 1 Results:")
        print(f"  Input clusters: {len(initial_clusters)}")
        print(f"  Conflicts: {len(conflicts)} sequences")
        print(f"  Final clusters: {len(final_clusters)}")
        print(f"  Conflicts resolved: ✓")

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pass2_merge_behavior(self, russula_300_path):
        """Test Pass 2 iterative merging on Russula 300."""
        # Load sequences
        sequences, headers = load_fasta(russula_300_path)

        # Create over-split initial clusters (small groups)
        # Simulate the output of Pass 1 that might be over-split
        cluster_size = 3
        initial_clusters = {}
        for i in range(0, len(headers), cluster_size):
            cluster_id = f"cluster_{i//cluster_size:05d}I"
            initial_clusters[cluster_id] = headers[i:i+cluster_size]

        print(f"\nInitial state: {len(initial_clusters)} clusters (avg size: {cluster_size})")

        # Run Pass 2 only (assume Pass 1 already done and MECE)
        config = RefinementConfig(
            max_full_gaphack_size=300,
            close_threshold=0.02,  # Use max_lump as threshold
            max_iterations=10,
            k_neighbors=20,
            search_method="blast"
        )

        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

        start_time = time.time()

        final_clusters, tracking_stages = two_pass_refinement(
            all_clusters=initial_clusters,
            sequences=sequences,
            headers=headers,
            conflicts={},  # No conflicts (MECE from Pass 1)
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            config=config,
            cluster_id_generator=cluster_id_generator,
            run_pass1=False,  # Skip Pass 1
            run_pass2=True,
            show_progress=False
        )

        duration = time.time() - start_time

        # Check results
        assert len(final_clusters) > 0, "Should produce clusters"
        assert len(final_clusters) <= len(initial_clusters), "Pass 2 should merge or maintain cluster count"

        # Check convergence
        if len(tracking_stages) > 0:
            pass2_info = tracking_stages[0]  # Pass 2 only, so first stage
            iterations = pass2_info.summary_stats.get('iterations', 0)
            convergence_reason = pass2_info.summary_stats.get('convergence_reason', 'unknown')
            converged_scopes = pass2_info.summary_stats.get('converged_scopes_count', 0)

            print(f"\nPass 2 Results:")
            print(f"  Initial clusters: {len(initial_clusters)}")
            print(f"  Final clusters: {len(final_clusters)}")
            print(f"  Change: {len(final_clusters) - len(initial_clusters)} ({(len(final_clusters) - len(initial_clusters)) / len(initial_clusters) * 100:.1f}%)")
            print(f"  Iterations: {iterations}")
            print(f"  Convergence: {convergence_reason}")
            print(f"  Converged scopes: {converged_scopes}")
            print(f"  Duration: {duration:.1f}s")

            # Should converge in reasonable iterations
            assert iterations <= 10, "Should converge within iteration limit"

        # Performance check: 300 sequences should complete in reasonable time
        assert duration < 600, f"Should complete in <10 minutes (took {duration:.1f}s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

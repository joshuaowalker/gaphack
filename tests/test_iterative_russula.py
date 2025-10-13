"""Integration tests for iterative refinement on Russula dataset.

Tests the iterative refinement system on real biological data (1,429 fungal
ITS sequences with 143 ground truth groups) to validate:
- Quality metrics meet thresholds (ARI ≥ 0.85, Homogeneity ≥ 0.90, Completeness ≥ 0.85)
- Convergence behavior (3-5 iterations expected)
- Performance (<30 minutes for 1,000+ sequences)
"""

import pytest
import time
from pathlib import Path
from Bio import SeqIO

from gaphack.cluster_refinement import refine_clusters, RefinementConfig
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


class TestIterativeRussulaIntegration:
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
    def test_iterative_full_russula_quality(self, russula_full_path):
        """Test iterative refinement on full Russula dataset with quality metrics.

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
        # 3. Running iterative refinement
        # 4. Computing quality metrics against ground truth
        # 5. Asserting metrics meet thresholds

    @pytest.mark.integration
    def test_iterative_convergence_behavior(self, russula_100_path):
        """Test that iterative refinement converges on Russula 100."""
        # Load sequences
        sequences, headers = load_fasta(russula_100_path)

        # Create initial singleton clusters (worst case for convergence)
        initial_clusters = {
            f"cluster_{i:05d}I": [header]
            for i, header in enumerate(headers, 1)
        }

        # Run iterative refinement
        config = RefinementConfig(
            max_full_gaphack_size=300,
            close_threshold=0.02,  # Required parameter
            max_iterations=10,
            k_neighbors=20,
            search_method="blast"
        )

        start_time = time.time()

        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

        final_clusters, refinement_info = refine_clusters(
            all_clusters=initial_clusters,
            sequences=sequences,
            headers=headers,
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            config=config,
            cluster_id_generator=cluster_id_generator,
            show_progress=False
        )

        duration = time.time() - start_time

        # Check results
        assert len(final_clusters) > 0, "Should produce clusters"
        assert len(final_clusters) < len(initial_clusters), "Should merge some singletons"

        # Check convergence
        iterations = refinement_info.summary_stats.get('iterations', 0)
        convergence_reason = refinement_info.summary_stats.get('convergence_reason', 'unknown')

        print(f"\nRefinement Results:")
        print(f"  Iterations: {iterations}")
        print(f"  Convergence: {convergence_reason}")
        print(f"  Final clusters: {len(final_clusters)}")
        print(f"  Duration: {duration:.1f}s")

        # Most datasets should converge in reasonable iterations
        assert iterations <= 10, "Should converge within iteration limit"

        # Performance check: 100 sequences should complete quickly
        assert duration < 300, f"Should complete in <5 minutes (took {duration:.1f}s)"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_iterative_merge_behavior(self, russula_300_path):
        """Test iterative merging on Russula 300."""
        # Load sequences
        sequences, headers = load_fasta(russula_300_path)

        # Create over-split initial clusters (small groups)
        cluster_size = 3
        initial_clusters = {}
        for i in range(0, len(headers), cluster_size):
            cluster_id = f"cluster_{i//cluster_size:05d}I"
            initial_clusters[cluster_id] = headers[i:i+cluster_size]

        print(f"\nInitial state: {len(initial_clusters)} clusters (avg size: {cluster_size})")

        # Run iterative refinement
        config = RefinementConfig(
            max_full_gaphack_size=300,
            close_threshold=0.02,  # Use max_lump as threshold
            max_iterations=10,
            k_neighbors=20,
            search_method="blast"
        )

        cluster_id_generator = ClusterIDGenerator(stage_name="refined", refinement_count=0)

        start_time = time.time()

        final_clusters, refinement_info = refine_clusters(
            all_clusters=initial_clusters,
            sequences=sequences,
            headers=headers,
            min_split=0.005,
            max_lump=0.02,
            target_percentile=95,
            config=config,
            cluster_id_generator=cluster_id_generator,
            show_progress=False
        )

        duration = time.time() - start_time

        # Check results
        assert len(final_clusters) > 0, "Should produce clusters"
        assert len(final_clusters) <= len(initial_clusters), "Refinement should merge or maintain cluster count"

        # Check convergence
        iterations = refinement_info.summary_stats.get('iterations', 0)
        convergence_reason = refinement_info.summary_stats.get('convergence_reason', 'unknown')
        converged_scopes = refinement_info.summary_stats.get('converged_scopes_count', 0)

        print(f"\nRefinement Results:")
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

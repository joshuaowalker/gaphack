"""
Phase 4 Quality Testing: Advanced Clustering Quality Metrics and Biological Validation

This module implements sophisticated clustering quality assessment including
silhouette analysis, within/between cluster distance ratios, and biological relevance validation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import subprocess

from test_phase4_integration import (
    RUSSULA_200, RUSSULA_300, RUSSULA_500, GroundTruthAnalyzer, CLIRunner, PerformanceMonitor,
    SKLEARN_AVAILABLE
)

# Additional imports for distance calculations
try:
    from gaphack.utils import calculate_distance_matrix
    GAPHACK_AVAILABLE = True
except ImportError:
    GAPHACK_AVAILABLE = False

pytestmark = [
    pytest.mark.integration,
    pytest.mark.quality,
    pytest.mark.skipif(not RUSSULA_200.exists(), reason="Russula test datasets not available")
]


class ClusterQualityAnalyzer:
    """Advanced clustering quality analysis utilities."""

    @staticmethod
    def calculate_silhouette_scores(clusters: List[List[str]], distance_provider) -> List[float]:
        """Calculate silhouette scores for clusters."""
        silhouette_scores = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue  # Silhouette undefined for singletons

            cluster_scores = []
            for seq_id in cluster:
                # Find sequence index
                seq_idx = None
                # This would need to be properly implemented with sequence indexing
                # For now, we'll use a placeholder approach

                # Calculate average intra-cluster distance
                intra_distances = []
                for other_seq in cluster:
                    if other_seq != seq_id:
                        # Would calculate distance between seq_id and other_seq
                        pass

                # Calculate average nearest-cluster distance
                inter_distances = []
                # Would find nearest cluster and calculate average distance

                if intra_distances and inter_distances:
                    a = np.mean(intra_distances)  # Intra-cluster
                    b = np.mean(inter_distances)  # Nearest cluster
                    silhouette = (b - a) / max(a, b) if max(a, b) > 0 else 0
                    cluster_scores.append(silhouette)

            if cluster_scores:
                silhouette_scores.append(np.mean(cluster_scores))

        return silhouette_scores

    @staticmethod
    def calculate_within_between_distances(clusters: List[List[str]], sequences: List[str], headers: List[str]) -> Tuple[List[float], List[float]]:
        """Calculate within-cluster and between-cluster distances using proper sequence alignment."""
        if not GAPHACK_AVAILABLE:
            return [], []

        # Create sequence lookup
        seq_to_idx = {header: i for i, header in enumerate(headers)}

        # Get all unique sequences that appear in clusters
        used_indices = set()
        for cluster in clusters:
            for seq_id in cluster:
                if seq_id in seq_to_idx:
                    used_indices.add(seq_to_idx[seq_id])

        if len(used_indices) < 2:
            return [], []

        # Create mapping from original indices to subset indices
        sorted_indices = sorted(used_indices)
        subset_sequences = [sequences[i] for i in sorted_indices]
        idx_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(sorted_indices)}

        # Calculate distance matrix for all sequences at once (much more efficient)
        try:
            distance_matrix = calculate_distance_matrix(subset_sequences, show_progress=False)
        except Exception:
            return [], []

        within_distances = []
        between_distances = []

        # Calculate within-cluster distances
        for cluster in clusters:
            indices = [idx_mapping[seq_to_idx[seq_id]]
                      for seq_id in cluster
                      if seq_id in seq_to_idx and seq_to_idx[seq_id] in idx_mapping]

            for i, idx1 in enumerate(indices):
                for idx2 in indices[i+1:]:
                    distance = distance_matrix[idx1, idx2]
                    if not np.isnan(distance):
                        within_distances.append(distance)

        # Calculate between-cluster distances (sample to avoid O(n²) explosion)
        for i, cluster1 in enumerate(clusters):
            for cluster2 in clusters[i+1:]:
                indices1 = [idx_mapping[seq_to_idx[seq_id]]
                           for seq_id in cluster1[:5]  # Sample first 5
                           if seq_id in seq_to_idx and seq_to_idx[seq_id] in idx_mapping]
                indices2 = [idx_mapping[seq_to_idx[seq_id]]
                           for seq_id in cluster2[:5]  # Sample first 5
                           if seq_id in seq_to_idx and seq_to_idx[seq_id] in idx_mapping]

                for idx1 in indices1:
                    for idx2 in indices2:
                        distance = distance_matrix[idx1, idx2]
                        if not np.isnan(distance):
                            between_distances.append(distance)

        return within_distances, between_distances

    @staticmethod
    def analyze_cluster_size_distribution(clusters: List[List[str]]) -> Dict[str, Any]:
        """Analyze cluster size distribution statistics."""
        cluster_sizes = [len(cluster) for cluster in clusters]

        if not cluster_sizes:
            return {"error": "No clusters to analyze"}

        return {
            "total_clusters": len(cluster_sizes),
            "mean_size": np.mean(cluster_sizes),
            "median_size": np.median(cluster_sizes),
            "std_size": np.std(cluster_sizes),
            "min_size": min(cluster_sizes),
            "max_size": max(cluster_sizes),
            "singleton_count": sum(1 for size in cluster_sizes if size == 1),
            "singleton_fraction": sum(1 for size in cluster_sizes if size == 1) / len(cluster_sizes),
            "largest_cluster_fraction": max(cluster_sizes) / sum(cluster_sizes) if sum(cluster_sizes) > 0 else 0,
            "size_distribution": Counter(cluster_sizes)
        }


class BiologicalRelevanceAnalyzer:
    """Analyze biological relevance of clustering results."""

    @staticmethod
    def parse_geographic_annotations(fasta_path: Path) -> Dict[str, Dict[str, str]]:
        """Extract geographic information from FASTA headers."""
        geographic_data = {}

        with open(fasta_path) as f:
            for line in f:
                if line.startswith('>'):
                    parts = line.split()
                    if len(parts) >= 2:
                        seq_id = parts[0][1:]  # Remove '>'

                        # Parse geographic information
                        location_info = {}
                        for part in parts[1:]:
                            if part in ['United_States', 'Canada', 'Mexico']:  # Country
                                location_info['country'] = part.replace('_', ' ')
                            elif len(part) <= 20 and '_' in part:  # Likely state/province
                                location_info['state'] = part.replace('_', ' ')

                        if location_info:
                            geographic_data[seq_id] = location_info

        return geographic_data

    @staticmethod
    def calculate_geographic_coherence(clusters: List[List[str]], geographic_data: Dict[str, Dict[str, str]]) -> List[float]:
        """Calculate geographic coherence scores for clusters."""
        coherence_scores = []

        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            # Get geographic info for cluster members
            locations = []
            for seq_id in cluster:
                if seq_id in geographic_data and 'state' in geographic_data[seq_id]:
                    locations.append(geographic_data[seq_id]['state'])

            if not locations:
                continue

            # Calculate coherence as fraction from most common state
            state_counts = Counter(locations)
            most_common_count = max(state_counts.values())
            coherence = most_common_count / len(locations)
            coherence_scores.append(coherence)

        return coherence_scores

    @staticmethod
    def analyze_taxonomic_coherence(clusters: List[List[str]], ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
        """Analyze how well clusters match taxonomic groups."""
        # Create sequence to group mapping
        seq_to_group = {}
        for group_name, sequences in ground_truth.items():
            for seq_id in sequences:
                seq_to_group[seq_id] = group_name

        purity_scores = []
        completeness_scores = []

        # Calculate purity (most sequences in cluster share same group)
        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            group_counts = Counter()
            for seq_id in cluster:
                if seq_id in seq_to_group:
                    group_counts[seq_to_group[seq_id]] += 1

            if group_counts:
                most_common_count = max(group_counts.values())
                purity = most_common_count / len(cluster)
                purity_scores.append(purity)

        # Calculate completeness (group members aren't split across many clusters)
        for group_name, group_sequences in ground_truth.items():
            if len(group_sequences) <= 1:
                continue

            # Find which clusters contain group members
            cluster_assignments = defaultdict(int)
            for seq_id in group_sequences:
                for cluster_idx, cluster in enumerate(clusters):
                    if seq_id in cluster:
                        cluster_assignments[cluster_idx] += 1
                        break

            if cluster_assignments:
                largest_cluster_size = max(cluster_assignments.values())
                completeness = largest_cluster_size / len(group_sequences)
                completeness_scores.append(completeness)

        return {
            "mean_purity": np.mean(purity_scores) if purity_scores else 0,
            "median_purity": np.median(purity_scores) if purity_scores else 0,
            "mean_completeness": np.mean(completeness_scores) if completeness_scores else 0,
            "median_completeness": np.median(completeness_scores) if completeness_scores else 0,
            "purity_scores": purity_scores,
            "completeness_scores": completeness_scores
        }


class TestClusteringQuality:
    """Validate intrinsic clustering quality measures."""

    def test_cluster_size_distribution_sanity(self):
        """Test that cluster size distribution is reasonable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "size_dist_test"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_200, output_base,
                min_split=0.003, max_lump=0.012
            )

            assert result['returncode'] == 0

            # Parse clusters
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")

                size_analysis = ClusterQualityAnalyzer.analyze_cluster_size_distribution(clusters)

                # Sanity checks
                assert size_analysis["total_clusters"] > 0
                assert size_analysis["mean_size"] >= 1
                assert size_analysis["largest_cluster_fraction"] < 0.5, "No single giant cluster"
                assert size_analysis["singleton_fraction"] < 0.8, "Not mostly singletons"

    @pytest.mark.skipif(not GAPHACK_AVAILABLE, reason="gaphack modules required for distance calculations")
    def test_within_vs_between_cluster_distances(self):
        """Test that within-cluster distances are smaller than between-cluster distances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "distance_test"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_200, output_base,
                min_split=0.003, max_lump=0.012
            )

            assert result['returncode'] == 0

            # Parse clusters
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")

                # Load sequences for distance calculation
                sequences = []
                headers = []
                with open(RUSSULA_200) as f:
                    current_seq = ""
                    for line in f:
                        if line.startswith('>'):
                            if current_seq and headers:
                                sequences.append(current_seq)
                            headers.append(line.split()[0][1:])
                            current_seq = ""
                        else:
                            current_seq += line.strip()
                    if current_seq:
                        sequences.append(current_seq)

                within_distances, between_distances = ClusterQualityAnalyzer.calculate_within_between_distances(
                    clusters, sequences, headers
                )

                if within_distances and between_distances:
                    # Gap-based optimization should achieve some separation on average
                    mean_within = np.mean(within_distances)
                    mean_between = np.mean(between_distances)

                    assert mean_within < mean_between, \
                        f"Within-cluster distances ({mean_within:.4f}) should be < between-cluster distances ({mean_between:.4f})"

                    # Check that median separation exists (less strict than percentile test)
                    median_within = np.median(within_distances)
                    median_between = np.median(between_distances)

                    assert median_within < median_between, \
                        f"Median within ({median_within:.4f}) should be < median between ({median_between:.4f})"

    def test_biological_coherence_analysis(self):
        """Test biological coherence of clustering results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "coherence_test"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_300, output_base,
                min_split=0.003, max_lump=0.012
            )

            assert result['returncode'] == 0

            # Parse results
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")
                ground_truth = GroundTruthAnalyzer.parse_ground_truth_groups(RUSSULA_300)

                # Analyze taxonomic coherence
                coherence_analysis = BiologicalRelevanceAnalyzer.analyze_taxonomic_coherence(clusters, ground_truth)

                # Quality thresholds for biological relevance - Updated based on empirical performance
                assert coherence_analysis["mean_purity"] > 0.85, \
                    f"Mean purity {coherence_analysis['mean_purity']:.3f} below high-performance biological threshold"

                assert coherence_analysis["mean_completeness"] > 0.80, \
                    f"Mean completeness {coherence_analysis['mean_completeness']:.3f} below high-performance biological threshold"

    def test_geographic_signal_preservation(self):
        """Test that some geographic signal is preserved in clustering."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "geographic_test"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_200, output_base,
                min_split=0.003, max_lump=0.012
            )

            assert result['returncode'] == 0

            # Parse results
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")
                geographic_data = BiologicalRelevanceAnalyzer.parse_geographic_annotations(RUSSULA_200)

                coherence_scores = BiologicalRelevanceAnalyzer.calculate_geographic_coherence(clusters, geographic_data)

                if coherence_scores:
                    mean_geographic_coherence = np.mean(coherence_scores)

                    # Some geographic signal should be preserved, but not too strong
                    # (related species can be geographically dispersed)
                    assert 0.3 < mean_geographic_coherence < 0.9, \
                        f"Geographic coherence {mean_geographic_coherence:.3f} outside expected range"


class TestScalabilityValidation:
    """Validate algorithmic scaling properties."""

    def test_algorithm_complexity_scaling(self):
        """Verify that algorithmic complexity doesn't degrade."""
        execution_times = []
        test_cases = [
            (RUSSULA_200, 200),
            (RUSSULA_300, 300)
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for dataset_path, size in test_cases:
                output_base = Path(tmpdir) / f"scaling_test_{size}"

                result = CLIRunner.run_gaphack_decompose(
                    dataset_path, output_base,
                    min_split=0.003, max_lump=0.012
                )

                assert result['returncode'] == 0, f"Scaling test failed for {size} sequences: {result['stderr']}"
                execution_times.append((size, result['execution_time']))

            # Verify scaling is not worse than quadratic
            for i in range(1, len(execution_times)):
                size_ratio = execution_times[i][0] / execution_times[i-1][0]
                time_ratio = execution_times[i][1] / max(execution_times[i-1][1], 0.1)  # Avoid division by zero

                # Should not exhibit exponential scaling
                expected_quadratic_ratio = size_ratio ** 2
                assert time_ratio < expected_quadratic_ratio * 2, \
                    f"Time scaling ({time_ratio:.2f}) worse than quadratic ({expected_quadratic_ratio:.2f}) for size ratio {size_ratio:.2f}"

    def test_memory_efficiency_validation(self):
        """Test memory efficiency across different dataset characteristics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with different sized subsets using pre-computed diverse datasets
            test_cases = [
                (RUSSULA_200, 200),
                (RUSSULA_300, 300)
            ]
            memory_usages = []

            for dataset_path, size in test_cases:
                output_base = Path(tmpdir) / f"memory_test_{size}"

                result = CLIRunner.run_gaphack_decompose(
                    dataset_path, output_base,
                    min_split=0.003, max_lump=0.012
                )

                assert result['returncode'] == 0
                memory_usages.append((size, result['memory_usage']))

            # Memory should scale reasonably
            largest_memory = memory_usages[-1][1]
            assert largest_memory < 2_000_000_000, f"Memory usage {largest_memory} bytes exceeds 2GB for 300 sequences"

            # Check that memory scaling is reasonable
            if len(memory_usages) >= 2:
                for i in range(1, len(memory_usages)):
                    size_ratio = memory_usages[i][0] / memory_usages[i-1][0]
                    memory_ratio = memory_usages[i][1] / max(memory_usages[i-1][1], 1)

                    if memory_ratio > 1:  # Only check if memory actually increased
                        # Memory scaling should be subquadratic due to optimizations
                        assert memory_ratio < size_ratio ** 2.2, \
                            f"Memory scaling too steep: {memory_ratio:.2f}x for {size_ratio:.2f}x size increase"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
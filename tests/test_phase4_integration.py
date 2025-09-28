"""
Phase 4 Integration Testing: End-to-End Workflows and Real Data Validation

This module implements comprehensive integration tests using the Russula_INxx.fasta dataset
to validate complete pipeline functionality, performance characteristics, and biological relevance.
"""

import os
import sys
import time
import json
import tempfile
import pytest
import subprocess
import gc
import resource  # Built-in module for memory monitoring
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import numpy as np
from unittest.mock import patch

# Import scikit-learn metrics for clustering validation
try:
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        adjusted_mutual_info_score,
        homogeneity_score,
        completeness_score,
        v_measure_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Test configuration
RUSSULA_DATASET = Path(__file__).parent.parent / "examples" / "data" / "Russula_INxx.fasta"
PERFORMANCE_BASELINES_FILE = Path(__file__).parent / "performance_baselines.json"

# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not RUSSULA_DATASET.exists(), reason="Russula dataset not available")
]


class PerformanceMonitor:
    """Monitor execution time, memory usage, and system resources."""

    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.peak_memory = None
        self.execution_time = None

    def __enter__(self):
        self.start_time = time.time()
        # Use resource module for memory monitoring (built-in)
        self.start_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.execution_time = time.time() - self.start_time
        self.peak_memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Note: ru_maxrss units vary by platform (bytes on Linux, KB on macOS)
        self.memory_growth = self.peak_memory - self.start_memory


class GroundTruthAnalyzer:
    """Utilities for analyzing clustering against biological ground truth."""

    @staticmethod
    def parse_ground_truth_groups(fasta_path: Path) -> Dict[str, List[str]]:
        """Extract name="" annotations from FASTA headers."""
        groups = defaultdict(list)

        with open(fasta_path) as f:
            for line in f:
                if line.startswith('>'):
                    seq_id = line.split()[0][1:]  # Remove '>'
                    if 'name="' in line:
                        start = line.find('name="') + 6
                        end = line.find('"', start)
                        if end > start:
                            group_name = line[start:end]
                            groups[group_name].append(seq_id)

        return dict(groups)

    @staticmethod
    def parse_clustering_results(output_path: Path, format_type: str = "tsv") -> List[List[str]]:
        """Parse clustering results from output files."""
        if format_type == "tsv":
            # Parse decompose TSV output format: sequence_id \t cluster_id
            cluster_dict = defaultdict(list)

            with open(output_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and not line.startswith("sequence_id"):
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            seq_id, cluster_id = parts[0], parts[1]
                            cluster_dict[cluster_id].append(seq_id)

            # Convert to list of clusters
            return list(cluster_dict.values())

        elif format_type == "fasta":
            # Parse FASTA cluster files
            clusters = []
            cluster_files = list(output_path.parent.glob(f"{output_path.stem}.cluster_*.fasta"))
            for cluster_file in sorted(cluster_files):
                cluster = []
                with open(cluster_file) as f:
                    for line in f:
                        if line.startswith('>'):
                            seq_id = line.split()[0][1:]
                            cluster.append(seq_id)
                if cluster:
                    clusters.append(cluster)
            return clusters

        return []

    @staticmethod
    def calculate_clustering_metrics(clusters: List[List[str]], ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate comprehensive clustering quality metrics."""
        if not SKLEARN_AVAILABLE:
            return {"error": "sklearn not available for metric calculation"}

        # Create sequence to cluster mapping
        seq_to_cluster = {}
        for cluster_idx, cluster in enumerate(clusters):
            for seq_id in cluster:
                seq_to_cluster[seq_id] = cluster_idx

        # Create sequence to ground truth mapping
        seq_to_truth = {}
        for group_name, sequences in ground_truth.items():
            for seq_id in sequences:
                seq_to_truth[seq_id] = group_name

        # Find common sequences
        common_sequences = set(seq_to_cluster.keys()) & set(seq_to_truth.keys())

        if not common_sequences:
            return {"error": "No common sequences between clustering and ground truth"}

        # Create label arrays
        cluster_labels = [seq_to_cluster[seq_id] for seq_id in common_sequences]
        true_labels = [seq_to_truth[seq_id] for seq_id in common_sequences]

        return {
            'adjusted_rand_index': adjusted_rand_score(true_labels, cluster_labels),
            'normalized_mutual_info': normalized_mutual_info_score(true_labels, cluster_labels),
            'adjusted_mutual_info': adjusted_mutual_info_score(true_labels, cluster_labels),
            'homogeneity': homogeneity_score(true_labels, cluster_labels),
            'completeness': completeness_score(true_labels, cluster_labels),
            'v_measure': v_measure_score(true_labels, cluster_labels),
            'common_sequences': len(common_sequences)
        }


class DatasetManager:
    """Manage test dataset creation and cleanup."""

    @staticmethod
    def create_subset(source_path: Path, n_sequences: int, output_path: Path = None) -> Path:
        """Create a subset of sequences from the source dataset."""
        if output_path is None:
            output_path = Path(tempfile.mktemp(suffix=f"_subset_{n_sequences}.fasta"))

        sequences_written = 0
        with open(source_path) as infile, open(output_path, 'w') as outfile:
            write_sequence = False
            for line in infile:
                if line.startswith('>'):
                    if sequences_written >= n_sequences:
                        break
                    write_sequence = True
                    sequences_written += 1

                if write_sequence:
                    outfile.write(line)

                # After writing sequence data, prepare for next header
                if write_sequence and not line.startswith('>') and line.strip():
                    write_sequence = False

        return output_path


class CLIRunner:
    """Run command-line tools and capture results."""

    @staticmethod
    def run_gaphack_decompose(input_path: Path, output_base: Path = None, **kwargs) -> Dict[str, Any]:
        """Run gaphack-decompose with specified parameters."""
        if output_base is None:
            output_base = Path(tempfile.mktemp(suffix="_decompose_output"))

        cmd = ['python', '-m', 'gaphack.decompose_cli', str(input_path), '-o', str(output_base)]

        # Add parameter flags
        for param, value in kwargs.items():
            param_name = param.replace('_', '-')
            if isinstance(value, bool) and value:
                cmd.append(f'--{param_name}')
            elif not isinstance(value, bool):
                cmd.extend([f'--{param_name}', str(value)])

        with PerformanceMonitor() as monitor:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'output_path': output_base,
            'execution_time': monitor.execution_time,
            'memory_usage': monitor.memory_growth
        }

    @staticmethod
    def run_gaphack_main(input_path: Path, output_base: Path = None, **kwargs) -> Dict[str, Any]:
        """Run main gaphack clustering tool."""
        if output_base is None:
            output_base = Path(tempfile.mktemp(suffix="_gaphack_output"))

        cmd = ['python', '-m', 'gaphack.cli', str(input_path), '-o', str(output_base)]

        # Add parameter flags
        for param, value in kwargs.items():
            param_name = param.replace('_', '-')
            if isinstance(value, bool) and value:
                cmd.append(f'--{param_name}')
            elif not isinstance(value, bool):
                cmd.extend([f'--{param_name}', str(value)])

        with PerformanceMonitor() as monitor:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent)

        return {
            'returncode': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'output_path': output_base,
            'execution_time': monitor.execution_time,
            'memory_usage': monitor.memory_growth
        }


class TestPipelineIntegration:
    """Test complete workflows from input to final analysis."""

    def test_decompose_full_dataset_runtime(self):
        """Test full Russula dataset completes in reasonable time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "russula_decompose"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_DATASET,
                output_base,
                min_split=0.003,
                max_lump=0.012,
                resolve_conflicts=True,
                refine_close_clusters=0.012
            )

            # Should complete successfully
            assert result['returncode'] == 0, f"Command failed: {result['stderr']}"

            # Should complete in reasonable time (<10 minutes)
            assert result['execution_time'] < 600, f"Execution time {result['execution_time']:.1f}s exceeds 10 minutes"

            # Should produce output files
            tsv_files = list(output_base.parent.glob(f"{output_base.name}*.tsv"))
            assert len(tsv_files) > 0, "No TSV output files produced"

    def test_gaphack_main_subset_functionality(self):
        """Test main gaphack tool with 300-sequence subset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create 300-sequence subset
            subset_path = Path(tmpdir) / "russula_subset_300.fasta"
            DatasetManager.create_subset(RUSSULA_DATASET, 300, subset_path)

            output_base = Path(tmpdir) / "russula_main"

            result = CLIRunner.run_gaphack_main(
                subset_path,
                output_base,
                min_split=0.003,
                max_lump=0.012,
                format="tsv"
            )

            # Should complete successfully
            assert result['returncode'] == 0, f"Command failed: {result['stderr']}"

            # Should complete in reasonable time (<5 minutes for subset)
            assert result['execution_time'] < 300, f"Execution time {result['execution_time']:.1f}s exceeds 5 minutes"

            # Should produce reasonable number of clusters
            output_file = output_base.with_suffix('.tsv')
            if output_file.exists():
                clusters = GroundTruthAnalyzer.parse_clustering_results(output_file, "tsv")
                assert 10 <= len(clusters) <= 100, f"Unexpected cluster count: {len(clusters)}"

    def test_output_format_consistency(self):
        """Test that different output formats produce consistent results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create small subset for quick testing
            subset_path = Path(tmpdir) / "russula_subset_100.fasta"
            DatasetManager.create_subset(RUSSULA_DATASET, 100, subset_path)

            # Run with TSV output
            tsv_output = Path(tmpdir) / "output_tsv"
            result_tsv = CLIRunner.run_gaphack_decompose(
                subset_path, tsv_output, min_split=0.003, max_lump=0.012
            )

            # Run with FASTA output
            fasta_output = Path(tmpdir) / "output_fasta"
            result_fasta = CLIRunner.run_gaphack_decompose(
                subset_path, fasta_output, min_split=0.003, max_lump=0.012, format="fasta"
            )

            # Both should succeed
            assert result_tsv['returncode'] == 0
            assert result_fasta['returncode'] == 0

            # Parse results
            tsv_file = next(tsv_output.parent.glob(f"{tsv_output.name}*.tsv"), None)
            if tsv_file:
                tsv_clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")
                fasta_clusters = GroundTruthAnalyzer.parse_clustering_results(fasta_output, "fasta")

                # Should produce similar number of clusters
                assert abs(len(tsv_clusters) - len(fasta_clusters)) <= 2, "Format outputs differ significantly"


class TestParameterSensitivity:
    """Validate algorithm behavior across parameter ranges."""

    @pytest.mark.parametrize("min_split,max_lump,expected_relation", [
        (0.001, 0.01, "restrictive"),
        (0.003, 0.012, "standard"),
        (0.005, 0.02, "permissive")
    ])
    def test_parameter_effects_on_clustering(self, min_split, max_lump, expected_relation):
        """Test various min-split/max-lump combinations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use medium subset for parameter testing
            subset_path = Path(tmpdir) / "russula_subset_200.fasta"
            DatasetManager.create_subset(RUSSULA_DATASET, 200, subset_path)

            output_base = Path(tmpdir) / f"param_test_{expected_relation}"

            result = CLIRunner.run_gaphack_decompose(
                subset_path, output_base,
                min_split=min_split,
                max_lump=max_lump
            )

            assert result['returncode'] == 0, f"Failed with {expected_relation} parameters"

            # Parse cluster count
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")

                # Store results for comparison (this would be enhanced with actual comparison logic)
                cluster_count = len(clusters)
                assert cluster_count > 0, f"No clusters produced with {expected_relation} parameters"

                # Basic sanity checks based on parameter strictness
                if expected_relation == "restrictive":
                    assert cluster_count >= 20, "Restrictive parameters should produce more clusters"
                elif expected_relation == "permissive":
                    assert cluster_count <= 50, "Permissive parameters should produce fewer clusters"


class TestGroundTruthValidation:
    """Validate clustering quality against manually curated groups."""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn required for clustering metrics")
    def test_clustering_quality_metrics(self):
        """Test clustering quality using standard metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use subset for faster testing
            subset_path = Path(tmpdir) / "russula_subset_500.fasta"
            DatasetManager.create_subset(RUSSULA_DATASET, 500, subset_path)

            output_base = Path(tmpdir) / "quality_test"

            result = CLIRunner.run_gaphack_decompose(
                subset_path, output_base,
                min_split=0.003, max_lump=0.012,
                resolve_conflicts=True
            )

            assert result['returncode'] == 0

            # Parse results
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")
                ground_truth = GroundTruthAnalyzer.parse_ground_truth_groups(subset_path)

                metrics = GroundTruthAnalyzer.calculate_clustering_metrics(clusters, ground_truth)

                # Quality thresholds - Updated based on empirical performance (ARI=0.948, Homogeneity=0.972, Completeness=0.969)
                assert metrics['adjusted_rand_index'] > 0.85, f"ARI {metrics['adjusted_rand_index']:.3f} below high-performance threshold"
                assert metrics['homogeneity'] > 0.90, f"Homogeneity {metrics['homogeneity']:.3f} below high-performance threshold"
                assert metrics['completeness'] > 0.85, f"Completeness {metrics['completeness']:.3f} below high-performance threshold"

    def test_cluster_count_reasonableness(self):
        """Validate reasonable cluster count for given parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            subset_path = Path(tmpdir) / "russula_subset_300.fasta"
            DatasetManager.create_subset(RUSSULA_DATASET, 300, subset_path)

            output_base = Path(tmpdir) / "count_test"

            result = CLIRunner.run_gaphack_decompose(
                subset_path, output_base,
                min_split=0.003, max_lump=0.012
            )

            assert result['returncode'] == 0

            # Parse cluster count
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")
                ground_truth = GroundTruthAnalyzer.parse_ground_truth_groups(subset_path)

                cluster_count = len(clusters)
                ground_truth_count = len(ground_truth)

                # Should produce reasonable number of clusters relative to ground truth
                assert 10 <= cluster_count <= ground_truth_count * 2, \
                    f"Cluster count {cluster_count} outside reasonable range vs {ground_truth_count} ground truth groups"


class TestPerformanceRegression:
    """Performance regression testing and baseline validation."""

    def test_performance_baseline_establishment(self):
        """Establish or validate performance baselines."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with multiple subset sizes
            test_sizes = [100, 300, 500]

            for size in test_sizes:
                subset_path = Path(tmpdir) / f"russula_subset_{size}.fasta"
                DatasetManager.create_subset(RUSSULA_DATASET, size, subset_path)

                output_base = Path(tmpdir) / f"perf_test_{size}"

                result = CLIRunner.run_gaphack_decompose(
                    subset_path, output_base,
                    min_split=0.003, max_lump=0.012
                )

                assert result['returncode'] == 0

                # Basic performance expectations
                if size == 100:
                    assert result['execution_time'] < 60, f"100 sequences took {result['execution_time']:.1f}s (expect <60s)"
                elif size == 300:
                    assert result['execution_time'] < 300, f"300 sequences took {result['execution_time']:.1f}s (expect <300s)"
                elif size == 500:
                    assert result['execution_time'] < 600, f"500 sequences took {result['execution_time']:.1f}s (expect <600s)"

    def test_memory_usage_scaling(self):
        """Test memory usage scaling with dataset size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test memory scaling
            test_sizes = [50, 100, 200]
            memory_usages = []

            for size in test_sizes:
                subset_path = Path(tmpdir) / f"russula_subset_{size}.fasta"
                DatasetManager.create_subset(RUSSULA_DATASET, size, subset_path)

                output_base = Path(tmpdir) / f"memory_test_{size}"

                result = CLIRunner.run_gaphack_decompose(
                    subset_path, output_base,
                    min_split=0.003, max_lump=0.012
                )

                assert result['returncode'] == 0
                memory_usages.append((size, result['memory_usage']))

            # Verify scaling is not worse than quadratic
            for i in range(1, len(memory_usages)):
                size_ratio = memory_usages[i][0] / memory_usages[i-1][0]
                memory_ratio = memory_usages[i][1] / max(memory_usages[i-1][1], 1)  # Avoid division by zero

                # Memory scaling should be reasonable (not exponential)
                if memory_ratio > 1:  # Only check if memory actually increased
                    assert memory_ratio < size_ratio ** 2.5, \
                        f"Memory scaling too steep: {memory_ratio:.2f}x for {size_ratio:.2f}x size increase"


# Helper functions for test setup and cleanup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment and cleanup after all tests."""
    # Ensure output directory exists
    os.makedirs(Path(__file__).parent / "test_outputs", exist_ok=True)

    yield

    # Cleanup any remaining temp files
    gc.collect()


if __name__ == "__main__":
    # Allow running individual test classes during development
    pytest.main([__file__, "-v"])
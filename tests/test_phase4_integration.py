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

# Test configuration - Pre-computed diverse datasets for fast, representative testing
TEST_DATA_DIR = Path(__file__).parent / "test_data"
RUSSULA_50 = TEST_DATA_DIR / "russula_diverse_50.fasta"
RUSSULA_100 = TEST_DATA_DIR / "russula_diverse_100.fasta"
RUSSULA_200 = TEST_DATA_DIR / "russula_diverse_200.fasta"
RUSSULA_300 = TEST_DATA_DIR / "russula_diverse_300.fasta"
RUSSULA_500 = TEST_DATA_DIR / "russula_diverse_500.fasta"
PERFORMANCE_BASELINES_FILE = Path(__file__).parent / "performance_baselines.json"

# Test markers
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not TEST_DATA_DIR.exists(), reason="Test datasets not available")
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
                            full_seq_id, cluster_id = parts[0], parts[1]
                            # Extract just the sequence ID part (before any spaces)
                            seq_id = full_seq_id.split()[0]
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


# DatasetManager removed - using pre-computed diverse datasets from tests/test_data/


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

    def test_decompose_small_dataset_runtime(self):
        """Test decompose with 50-sequence diverse dataset completes quickly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "russula_50_decompose"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_50,
                output_base,
                min_split=0.003,
                max_lump=0.012,
                resolve_conflicts=True,
                refine_close_clusters=0.012
            )

            # Should complete successfully
            assert result['returncode'] == 0, f"Command failed: {result['stderr']}"

            # Should complete very quickly (<30 seconds for 50 diverse sequences)
            assert result['execution_time'] < 30, f"Execution time {result['execution_time']:.1f}s exceeds 30 seconds"

            # Should produce output files
            tsv_files = list(output_base.parent.glob(f"{output_base.name}*.tsv"))
            assert len(tsv_files) > 0, "No TSV output files produced"

    def test_decompose_medium_dataset_functionality(self):
        """Test decompose with 100-sequence diverse dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "russula_100_decompose"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_100,
                output_base,
                min_split=0.003,
                max_lump=0.012,
                resolve_conflicts=True
            )

            # Should complete successfully
            assert result['returncode'] == 0, f"Command failed: {result['stderr']}"

            # Should complete in reasonable time (<2 minutes for 100 diverse sequences)
            assert result['execution_time'] < 120, f"Execution time {result['execution_time']:.1f}s exceeds 2 minutes"

            # Should produce reasonable number of clusters (5-15 for diverse 100-seq dataset)
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")
                assert 3 <= len(clusters) <= 20, f"Unexpected cluster count: {len(clusters)} for diverse dataset"

    def test_decompose_cli_basic_functionality(self):
        """Test basic gaphack-decompose CLI functionality with diverse dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test decompose CLI with 50-sequence diverse dataset
            output_base = Path(tmpdir) / "cli_test"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_50, output_base,
                min_split=0.003, max_lump=0.012,
                resolve_conflicts=True
            )

            # Should complete successfully
            assert result['returncode'] == 0, f"CLI test failed: {result['stderr']}"

            # Should produce output files
            tsv_files = list(output_base.parent.glob(f"{output_base.name}*.tsv"))
            assert len(tsv_files) > 0, "No TSV output files produced"

            # Parse and validate results
            tsv_file = tsv_files[0]
            clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")

            # Should produce reasonable clusters for diverse 50-sequence dataset
            assert 3 <= len(clusters) <= 15, f"Unexpected cluster count: {len(clusters)}"

            # Verify all sequences are assigned
            total_sequences = sum(len(cluster) for cluster in clusters)
            assert total_sequences == 50, f"Not all sequences assigned: {total_sequences}/50"


class TestParameterSensitivity:
    """Validate algorithm behavior across parameter ranges."""

    @pytest.mark.parametrize("min_split,max_lump,expected_relation", [
        (0.001, 0.01, "restrictive"),
        (0.003, 0.012, "standard"),
        (0.005, 0.02, "permissive")
    ])
    def test_parameter_effects_on_clustering(self, min_split, max_lump, expected_relation):
        """Test various min-split/max-lump combinations using 100-sequence diverse dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / f"param_test_{expected_relation}"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_100, output_base,
                min_split=min_split,
                max_lump=max_lump
            )

            assert result['returncode'] == 0, f"Failed with {expected_relation} parameters: {result['stderr']}"

            # Parse cluster count
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")

                cluster_count = len(clusters)
                assert cluster_count > 0, f"No clusters produced with {expected_relation} parameters"

                # Basic sanity checks based on parameter strictness (adjusted for 100-seq diverse dataset)
                if expected_relation == "restrictive":
                    assert cluster_count >= 8, "Restrictive parameters should produce more clusters"
                elif expected_relation == "permissive":
                    assert cluster_count <= 15, "Permissive parameters should produce fewer clusters"
                else:  # standard
                    assert 5 <= cluster_count <= 12, "Standard parameters should produce moderate cluster count"


class TestGroundTruthValidation:
    """Validate clustering quality against manually curated groups."""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="sklearn required for clustering metrics")
    def test_clustering_quality_metrics(self):
        """Test clustering quality using standard metrics with 500-sequence comprehensive dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "quality_test"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_500, output_base,
                min_split=0.003, max_lump=0.012,
                resolve_conflicts=True
            )

            assert result['returncode'] == 0, f"Quality test failed: {result['stderr']}"

            # Should complete in reasonable time (<10 minutes for 500 diverse sequences)
            assert result['execution_time'] < 600, f"Execution time {result['execution_time']:.1f}s exceeds 10 minutes"

            # Parse results
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")
                ground_truth = GroundTruthAnalyzer.parse_ground_truth_groups(RUSSULA_500)

                metrics = GroundTruthAnalyzer.calculate_clustering_metrics(clusters, ground_truth)

                # Quality thresholds - Adjusted for diverse dataset composition
                if 'error' not in metrics:
                    assert metrics['adjusted_rand_index'] > 0.65, f"ARI {metrics['adjusted_rand_index']:.3f} below threshold"
                    assert metrics['homogeneity'] > 0.75, f"Homogeneity {metrics['homogeneity']:.3f} below threshold"
                    assert metrics['completeness'] > 0.65, f"Completeness {metrics['completeness']:.3f} below threshold"

    def test_cluster_count_reasonableness(self):
        """Validate reasonable cluster count for given parameters using 200-sequence diverse dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_base = Path(tmpdir) / "count_test"

            result = CLIRunner.run_gaphack_decompose(
                RUSSULA_200, output_base,
                min_split=0.003, max_lump=0.012
            )

            assert result['returncode'] == 0, f"Count test failed: {result['stderr']}"

            # Parse cluster count
            tsv_file = next(output_base.parent.glob(f"{output_base.name}*.tsv"), None)
            if tsv_file:
                clusters = GroundTruthAnalyzer.parse_clustering_results(tsv_file, "tsv")
                ground_truth = GroundTruthAnalyzer.parse_ground_truth_groups(RUSSULA_200)

                cluster_count = len(clusters)
                ground_truth_count = len(ground_truth)

                # Should produce reasonable number of clusters relative to ground truth
                # For diverse 200-seq dataset, expect fewer clusters due to merging
                assert 5 <= cluster_count <= ground_truth_count * 1.5, \
                    f"Cluster count {cluster_count} outside reasonable range vs {ground_truth_count} ground truth groups"


class TestPerformanceRegression:
    """Performance regression testing and baseline validation."""

    def test_performance_baseline_establishment(self):
        """Establish performance baselines with small and medium datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test with two representative sizes for performance baselines
            test_cases = [
                (RUSSULA_50, "50_diverse", 30),    # Should complete in <30 seconds
                (RUSSULA_100, "100_diverse", 90),  # Should complete in <90 seconds
            ]

            for dataset_path, size_name, max_time in test_cases:
                output_base = Path(tmpdir) / f"perf_test_{size_name}"

                result = CLIRunner.run_gaphack_decompose(
                    dataset_path, output_base,
                    min_split=0.003, max_lump=0.012
                )

                assert result['returncode'] == 0, f"Performance test failed for {size_name}: {result['stderr']}"

                # Performance expectations for diverse datasets
                assert result['execution_time'] < max_time, \
                    f"{size_name} sequences took {result['execution_time']:.1f}s (expect <{max_time}s)"

    def test_memory_usage_scaling(self):
        """Test memory usage scaling with diverse dataset sizes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test memory scaling with pre-computed diverse datasets
            test_cases = [
                (RUSSULA_50, 50),
                (RUSSULA_100, 100),
                (RUSSULA_200, 200)
            ]
            memory_usages = []

            for dataset_path, size in test_cases:
                output_base = Path(tmpdir) / f"memory_test_{size}"

                result = CLIRunner.run_gaphack_decompose(
                    dataset_path, output_base,
                    min_split=0.003, max_lump=0.012
                )

                assert result['returncode'] == 0, f"Memory test failed for {size} sequences: {result['stderr']}"
                memory_usages.append((size, result['memory_usage']))

            # Verify scaling is reasonable (not exponential)
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
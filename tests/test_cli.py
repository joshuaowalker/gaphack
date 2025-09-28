"""
Tests for command-line interfaces.
"""

import pytest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

from gaphack.cli import main as cli_main, setup_logging
from gaphack.decompose_cli import main as decompose_main
from gaphack.analyze_cli import main as analyze_main


class TestMainCLI:
    """Test suite for main gapHACk CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = [
            "ATGCGATCGATCGATCG",
            "ATGCGATCGATCGATCC",
            "TTTTTTTTTTTTTTTTTT",
            "TTTTTTTTTTTTTTTTCC"
        ]

    def _create_test_fasta(self, sequences, filepath):
        """Helper to create test FASTA file."""
        with open(filepath, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")

    def test_setup_logging_verbose(self):
        """Test logging setup with verbose mode."""
        with patch('gaphack.cli.logging.basicConfig') as mock_config:
            setup_logging(verbose=True)
            mock_config.assert_called_once()
            args, kwargs = mock_config.call_args
            assert kwargs['level'] == 10  # logging.DEBUG

    def test_setup_logging_normal(self):
        """Test logging setup with normal mode."""
        with patch('gaphack.cli.logging.basicConfig') as mock_config:
            setup_logging(verbose=False)
            mock_config.assert_called_once()
            args, kwargs = mock_config.call_args
            assert kwargs['level'] == 20  # logging.INFO

    @patch('sys.argv', ['gaphack', '--help'])
    def test_cli_help_message(self):
        """Test that CLI shows help message."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main()
        # Help should exit with code 0
        assert exc_info.value.code == 0

    @patch('sys.argv', ['gaphack', 'nonexistent.fasta'])
    def test_cli_missing_input_file(self):
        """Test CLI behavior with missing input file."""
        with pytest.raises(SystemExit):
            cli_main()

    def test_cli_basic_clustering(self):
        """Test basic clustering functionality through CLI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test input file
            input_fasta = tmpdir / "test_input.fasta"
            self._create_test_fasta(self.test_sequences, input_fasta)

            # Mock sys.argv for CLI
            test_args = ['gaphack', str(input_fasta), '-o', str(tmpdir / 'output')]

            with patch('sys.argv', test_args), \
                 patch('gaphack.cli.GapOptimizedClustering') as mock_clustering, \
                 patch('gaphack.cli.load_sequences_from_fasta') as mock_load, \
                 patch('gaphack.cli.calculate_distance_matrix') as mock_calc_dist, \
                 patch('gaphack.cli.save_clusters_to_file') as mock_save:

                # Set up mocks
                mock_load.return_value = (
                    self.test_sequences,
                    [f"seq_{i}" for i in range(len(self.test_sequences))],
                    {f"seq_{i}": f"seq_{i}" for i in range(len(self.test_sequences))}
                )
                mock_calc_dist.return_value = [[0.0, 0.1], [0.1, 0.0]]

                mock_clustering_instance = Mock()
                mock_clustering_instance.cluster.return_value = ([[0, 1], [2, 3]], [], {})
                mock_clustering.return_value = mock_clustering_instance

                # Test should complete without error
                try:
                    cli_main()
                except SystemExit as e:
                    # Exit code 0 is success
                    assert e.code == 0 or e.code is None

    def test_cli_parameter_validation(self):
        """Test CLI parameter validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_fasta = tmpdir / "test_input.fasta"
            self._create_test_fasta(self.test_sequences, input_fasta)

            # Test invalid min-split parameter
            test_args = ['gaphack', str(input_fasta), '--min-split', 'invalid']

            with patch('sys.argv', test_args):
                with pytest.raises(SystemExit):
                    cli_main()

    def test_cli_target_mode(self):
        """Test CLI target mode functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_fasta = tmpdir / "input.fasta"
            target_fasta = tmpdir / "targets.fasta"
            output_file = tmpdir / "output.tsv"

            self._create_test_fasta(self.test_sequences, input_fasta)
            self._create_test_fasta(self.test_sequences[:2], target_fasta)

            test_args = ['gaphack', str(input_fasta), '--target', str(target_fasta), '-o', str(output_file)]

            with patch('sys.argv', test_args), \
                 patch('gaphack.cli.TargetModeClustering') as mock_target_clustering, \
                 patch('gaphack.cli.load_sequences_from_fasta') as mock_load, \
                 patch('gaphack.cli.DistanceProviderFactory') as mock_factory, \
                 patch('gaphack.cli.save_clusters_to_file') as mock_save, \
                 patch('gaphack.cli.validate_sequences') as mock_validate:

                # Mock sequence validation to return valid
                mock_validate.return_value = (True, [])

                # Mock load_sequences_from_fasta to return 3 values (sequences, headers, header_mapping)
                mock_load.return_value = (
                    self.test_sequences,
                    [f"seq_{i}" for i in range(len(self.test_sequences))],
                    {f"seq_{i}": f"seq_{i}" for i in range(len(self.test_sequences))}
                )

                # Mock distance provider
                mock_provider = Mock()
                mock_provider.get_cache_stats.return_value = {
                    'cached_distances': 10,
                    'theoretical_max': 100
                }
                mock_factory.create_lazy_provider.return_value = mock_provider

                # Mock target clustering with correct signature
                mock_target_instance = Mock()
                # cluster(distance_provider, target_indices, sequences) -> (target_cluster, remaining_sequences, metrics)
                mock_target_instance.cluster.return_value = ([0, 1], [2, 3], {'gap_size': 0.05})
                mock_target_clustering.return_value = mock_target_instance

                try:
                    cli_main()
                except SystemExit as e:
                    assert e.code == 0 or e.code is None

                # Verify target clustering was called correctly
                mock_target_instance.cluster.assert_called_once()
                args, kwargs = mock_target_instance.cluster.call_args
                assert len(args) == 3  # distance_provider, target_indices, sequences

    def test_cli_metrics_export(self):
        """Test CLI metrics export functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_fasta = tmpdir / "input.fasta"
            metrics_file = tmpdir / "metrics.json"

            self._create_test_fasta(self.test_sequences, input_fasta)

            test_args = ['gaphack', str(input_fasta), '--export-metrics', str(metrics_file)]

            with patch('sys.argv', test_args), \
                 patch('gaphack.cli.GapOptimizedClustering') as mock_clustering, \
                 patch('gaphack.cli.load_sequences_from_fasta') as mock_load, \
                 patch('gaphack.cli.calculate_distance_matrix') as mock_calc_dist, \
                 patch('gaphack.cli.save_clusters_to_file') as mock_save:

                mock_load.return_value = (
                    self.test_sequences,
                    [f"seq_{i}" for i in range(len(self.test_sequences))],
                    {f"seq_{i}": f"seq_{i}" for i in range(len(self.test_sequences))}
                )
                mock_calc_dist.return_value = [[0.0, 0.1], [0.1, 0.0]]

                test_metrics = {'gap_size': 0.05, 'clusters': 2}
                mock_clustering_instance = Mock()
                mock_clustering_instance.cluster.return_value = ([[0, 1]], [], test_metrics)
                mock_clustering.return_value = mock_clustering_instance

                with patch('builtins.open', mock_open()) as mock_file, \
                     patch('json.dump') as mock_json_dump:

                    try:
                        cli_main()
                    except SystemExit as e:
                        assert e.code == 0 or e.code is None

                    # Verify metrics were attempted to be written
                    mock_json_dump.assert_called()


class TestDecomposeCLI:
    """Test suite for decompose CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_sequences = [
            "ATGCGATCGATCGATCG",
            "ATGCGATCGATCGATCC",
            "TTTTTTTTTTTTTTTTTT"
        ]

    def _create_test_fasta(self, sequences, filepath):
        """Helper to create test FASTA file."""
        with open(filepath, 'w') as f:
            for i, seq in enumerate(sequences):
                f.write(f">seq_{i}\n{seq}\n")

    @patch('sys.argv', ['gaphack-decompose', '--help'])
    def test_decompose_cli_help(self):
        """Test decompose CLI help message."""
        with pytest.raises(SystemExit) as exc_info:
            decompose_main()
        assert exc_info.value.code == 0

    def test_decompose_cli_basic_functionality(self):
        """Test basic decompose CLI functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_fasta = tmpdir / "input.fasta"
            self._create_test_fasta(self.test_sequences, input_fasta)

            test_args = ['gaphack-decompose', str(input_fasta), '-o', str(tmpdir / 'output')]

            with patch('sys.argv', test_args), \
                 patch('gaphack.decompose_cli.DecomposeClustering') as mock_decompose, \
                 patch('gaphack.decompose_cli.save_decompose_results') as mock_save:

                # Mock decompose results
                mock_results = Mock()
                mock_results.clusters = {'cluster_1': ['seq_0'], 'cluster_2': ['seq_1', 'seq_2']}
                mock_results.conflicts = {}
                mock_results.unassigned = []

                mock_decompose_instance = Mock()
                mock_decompose_instance.decompose.return_value = mock_results
                mock_decompose.return_value = mock_decompose_instance

                try:
                    decompose_main()
                except SystemExit as e:
                    assert e.code == 0 or e.code is None

    def test_decompose_cli_with_targets(self):
        """Test decompose CLI with target sequences."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_fasta = tmpdir / "input.fasta"
            targets_fasta = tmpdir / "targets.fasta"

            self._create_test_fasta(self.test_sequences, input_fasta)
            self._create_test_fasta(self.test_sequences[:1], targets_fasta)

            test_args = ['gaphack-decompose', str(input_fasta), '--targets', str(targets_fasta), '-o', str(tmpdir / 'output')]

            with patch('sys.argv', test_args), \
                 patch('gaphack.decompose_cli.DecomposeClustering') as mock_decompose, \
                 patch('gaphack.decompose_cli.save_decompose_results') as mock_save:

                mock_results = Mock()
                mock_results.clusters = {'cluster_1': ['seq_0', 'seq_1']}
                mock_results.conflicts = {}
                mock_results.unassigned = ['seq_2']

                mock_decompose_instance = Mock()
                mock_decompose_instance.decompose.return_value = mock_results
                mock_decompose.return_value = mock_decompose_instance

                try:
                    decompose_main()
                except SystemExit as e:
                    assert e.code == 0 or e.code is None

    def test_decompose_cli_parameter_validation(self):
        """Test decompose CLI parameter validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            input_fasta = tmpdir / "input.fasta"
            self._create_test_fasta(self.test_sequences, input_fasta)

            # Test invalid max-clusters parameter
            test_args = ['gaphack-decompose', str(input_fasta), '--max-clusters', 'invalid']

            with patch('sys.argv', test_args):
                with pytest.raises(SystemExit):
                    decompose_main()


class TestAnalyzeCLI:
    """Test suite for analyze CLI."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_data = {
            'cluster_1': ['seq_0', 'seq_1'],
            'cluster_2': ['seq_2']
        }

    @patch('sys.argv', ['gaphack-analyze', '--help'])
    def test_analyze_cli_help(self):
        """Test analyze CLI help message."""
        with pytest.raises(SystemExit) as exc_info:
            analyze_main()
        assert exc_info.value.code == 0

    def test_analyze_cli_basic_functionality(self):
        """Test basic analyze CLI functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test FASTA files (analyze CLI expects FASTA files, not TSV)
            cluster_file1 = tmpdir / "cluster1.fasta"
            cluster_file2 = tmpdir / "cluster2.fasta"

            with open(cluster_file1, 'w') as f:
                f.write(">seq_0\nATGCGATCGATCGATC\n>seq_1\nATGCGATCGATCGATG\n")

            with open(cluster_file2, 'w') as f:
                f.write(">seq_2\nTTTTTTTTTTTTTTTT\n")

            output_dir = tmpdir / "analysis"

            test_args = ['gaphack-analyze', str(cluster_file1), str(cluster_file2), '-o', str(output_dir)]

            with patch('sys.argv', test_args), \
                 patch('gaphack.analyze_cli.analyze_single_cluster') as mock_analyze:

                # Mock analysis function to return expected structure
                mock_analyze.return_value = {
                    'filename': 'test_cluster.fasta',
                    'n_sequences': 3,
                    'n_distances': 3,
                    'distances': [0.1, 0.2, 0.3],
                    'percentiles': {
                        'P5': 0.1, 'P25': 0.15, 'P50': 0.2, 'P75': 0.25, 'P95': 0.3
                    },
                    'sequences': ['ATGC', 'GCTA', 'CGAT'],
                    'headers': ['seq_0', 'seq_1', 'seq_2']
                }

                try:
                    analyze_main()
                except SystemExit as e:
                    assert e.code == 0 or e.code is None

    def test_analyze_cli_missing_input(self):
        """Test analyze CLI with missing input file."""
        test_args = ['gaphack-analyze', 'nonexistent.tsv']

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                analyze_main()


# Import helper function for mock_open
try:
    from unittest.mock import mock_open
except ImportError:
    # For older Python versions
    def mock_open(mock=None, read_data=''):
        """Mock implementation of open() for testing."""
        if mock is None:
            mock = MagicMock()

        handle = MagicMock()
        handle.write.return_value = None
        handle.__enter__.return_value = handle
        handle.__exit__.return_value = False
        handle.read.return_value = read_data

        mock.return_value = handle
        return mock
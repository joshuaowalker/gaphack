"""
Tests for analysis functions.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from gaphack.analyze import (
    calculate_intra_cluster_distances,
    calculate_inter_cluster_distances,
    calculate_percentiles,
    calculate_barcode_gap_metrics,
    create_histogram,
    create_combined_histogram,
    format_analysis_report
)


class TestDistanceAnalysis:
    """Test distance calculation functions."""
    
    def test_calculate_intra_cluster_distances(self):
        """Test intra-cluster distance calculation."""
        sequences = ["ATCG", "ATCC", "TACG", "TACG"]  # 4 sequences
        distances = calculate_intra_cluster_distances(sequences)
        
        # Should have 6 pairwise distances (4 choose 2)
        assert len(distances) == 6
        assert all(0.0 <= d <= 1.0 for d in distances)
    
    def test_calculate_intra_cluster_distances_single_sequence(self):
        """Test intra-cluster distances with single sequence."""
        sequences = ["ATCG"]
        distances = calculate_intra_cluster_distances(sequences)
        
        # Should have no distances for single sequence
        assert len(distances) == 0
    
    def test_calculate_intra_cluster_distances_empty(self):
        """Test intra-cluster distances with empty sequences."""
        sequences = []
        distances = calculate_intra_cluster_distances(sequences)
        
        # Should have no distances for empty list
        assert len(distances) == 0
    
    def test_calculate_inter_cluster_distances(self):
        """Test inter-cluster distance calculation."""
        cluster_sequences = [
            ["ATCG", "ATCC"],  # Cluster 1: 2 sequences
            ["TACG", "TACC"]   # Cluster 2: 2 sequences
        ]
        distances = calculate_inter_cluster_distances(cluster_sequences)
        
        # Should have 4 inter-cluster distances (2*2)
        assert len(distances) == 4
        assert all(0.0 <= d <= 1.0 for d in distances)
    
    def test_calculate_inter_cluster_distances_single_cluster(self):
        """Test inter-cluster distances with single cluster."""
        cluster_sequences = [["ATCG", "ATCC"]]
        distances = calculate_inter_cluster_distances(cluster_sequences)
        
        # Should have no inter-cluster distances with only one cluster
        assert len(distances) == 0
    
    def test_calculate_inter_cluster_distances_empty(self):
        """Test inter-cluster distances with empty clusters."""
        cluster_sequences = []
        distances = calculate_inter_cluster_distances(cluster_sequences)
        
        # Should have no distances for empty clusters
        assert len(distances) == 0


class TestPercentileCalculations:
    """Test percentile calculation functions."""
    
    def test_calculate_percentiles(self):
        """Test percentile calculation."""
        distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        percentiles = calculate_percentiles(distances)
        
        # Check that all expected percentiles are present
        expected_keys = ['P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95']
        assert all(key in percentiles for key in expected_keys)
        
        # Check that percentiles are in ascending order
        values = [percentiles[key] for key in expected_keys]
        assert all(values[i] <= values[i+1] for i in range(len(values)-1))
        
        # Check median is correct
        assert percentiles['P50'] == 0.5
    
    def test_calculate_percentiles_empty(self):
        """Test percentile calculation with empty array."""
        distances = np.array([])
        percentiles = calculate_percentiles(distances)
        
        # Should return NaN for all percentiles
        assert all(np.isnan(value) for value in percentiles.values())


class TestBarcodeGapMetrics:
    """Test barcode gap calculation functions."""
    
    def test_calculate_barcode_gap_metrics(self):
        """Test barcode gap metrics calculation."""
        # Create clear gap scenario
        intra_distances = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # Intra-cluster: 0.01-0.05
        inter_distances = np.array([0.10, 0.11, 0.12, 0.13, 0.14])  # Inter-cluster: 0.10-0.14
        
        gap_metrics = calculate_barcode_gap_metrics(intra_distances, inter_distances)
        
        # Should detect a gap for both P90 and P95
        assert 'P90' in gap_metrics
        assert 'P95' in gap_metrics
        
        for percentile in ['P90', 'P95']:
            metrics = gap_metrics[percentile]
            assert 'intra_percentile' in metrics
            assert 'inter_percentile' in metrics
            assert 'gap_size' in metrics
            assert 'gap_exists' in metrics
            
            # Should have a positive gap
            assert metrics['gap_exists']
            assert metrics['gap_size'] > 0
    
    def test_calculate_barcode_gap_metrics_no_gap(self):
        """Test barcode gap metrics with overlapping distributions."""
        # Create overlapping scenario
        intra_distances = np.array([0.05, 0.06, 0.07, 0.08, 0.09])  # Intra-cluster: 0.05-0.09
        inter_distances = np.array([0.06, 0.07, 0.08, 0.09, 0.10])  # Inter-cluster: 0.06-0.10
        
        gap_metrics = calculate_barcode_gap_metrics(intra_distances, inter_distances)
        
        # May or may not have a gap depending on percentiles
        for percentile in ['P90', 'P95']:
            metrics = gap_metrics[percentile]
            assert 'gap_size' in metrics
            assert 'gap_exists' in metrics
    
    def test_calculate_barcode_gap_metrics_empty_data(self):
        """Test barcode gap metrics with empty data."""
        intra_distances = np.array([])
        inter_distances = np.array([0.10, 0.11])
        
        gap_metrics = calculate_barcode_gap_metrics(intra_distances, inter_distances)
        
        # Should return NaN values and no gap
        for percentile in ['P90', 'P95']:
            metrics = gap_metrics[percentile]
            assert np.isnan(metrics['gap_size'])
            assert not metrics['gap_exists']


class TestVisualization:
    """Test visualization functions."""
    
    def test_create_histogram(self):
        """Test histogram creation."""
        distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        fig = create_histogram(distances, "Test Histogram")
        
        # Check that figure was created
        assert fig is not None
        
        # Check that it has the expected structure
        axes = fig.get_axes()
        assert len(axes) == 1
        
        plt.close(fig)
    
    def test_create_histogram_empty(self):
        """Test histogram creation with empty data."""
        distances = np.array([])
        
        fig = create_histogram(distances, "Empty Histogram")
        
        # Should still create a figure
        assert fig is not None
        plt.close(fig)
    
    def test_create_combined_histogram(self):
        """Test combined histogram creation."""
        intra_distances = np.array([0.01, 0.02, 0.03])
        inter_distances = np.array([0.10, 0.11, 0.12])
        
        fig = create_combined_histogram(intra_distances, inter_distances)
        
        # Check that figure was created
        assert fig is not None
        
        # Check that it has the expected structure
        axes = fig.get_axes()
        assert len(axes) == 1
        
        plt.close(fig)
    
    def test_create_combined_histogram_empty(self):
        """Test combined histogram with empty data."""
        intra_distances = np.array([])
        inter_distances = np.array([])
        
        fig = create_combined_histogram(intra_distances, inter_distances)
        
        # Should still create a figure
        assert fig is not None
        plt.close(fig)


class TestReportFormatting:
    """Test report formatting functions."""
    
    def test_format_analysis_report(self):
        """Test analysis report formatting."""
        cluster_stats = [
            {
                'filename': 'cluster1.fasta',
                'n_sequences': 10,
                'n_distances': 45,
                'percentiles': {'P5': 0.01, 'P50': 0.05, 'P95': 0.10}
            },
            {
                'filename': 'cluster2.fasta',
                'n_sequences': 5,
                'n_distances': 10,
                'percentiles': {'P5': 0.02, 'P50': 0.06, 'P95': 0.12}
            }
        ]
        
        global_stats = {
            'total_sequences': 15,
            'n_intra': 55,
            'n_inter': 50,
            'intra_percentiles': {'P5': 0.01, 'P50': 0.05, 'P95': 0.11},
            'inter_percentiles': {'P5': 0.15, 'P50': 0.20, 'P95': 0.25}
        }
        
        gap_metrics = {
            'P95': {
                'intra_percentile': 0.11,
                'inter_percentile': 0.15,
                'gap_size': 0.04,
                'gap_exists': True
            }
        }
        
        report = format_analysis_report(cluster_stats, global_stats, gap_metrics)
        
        # Check that report contains expected sections
        assert "INDIVIDUAL CLUSTER ANALYSIS" in report
        assert "GLOBAL ANALYSIS" in report
        assert "BARCODE GAP ANALYSIS" in report
        assert "cluster1.fasta" in report
        assert "cluster2.fasta" in report
        assert "Total sequences: 15" in report
        assert "Gap exists: Yes" in report
"""
Analysis functions for pre-clustered FASTA files.

This module provides functionality to analyze distance distributions and barcode gaps
from FASTA files that are assumed to contain pre-clustered sequences.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import logging
from .utils import load_sequences_from_fasta, calculate_distance_matrix


def calculate_intra_cluster_distances(sequences: List[str], 
                                    alignment_method: str = "adjusted",
                                    **alignment_kwargs) -> np.ndarray:
    """
    Calculate all pairwise distances within a cluster.
    
    Args:
        sequences: List of DNA sequences
        alignment_method: Method for distance calculation
        **alignment_kwargs: Additional arguments for distance calculation
        
    Returns:
        1D array of pairwise distances (upper triangle, excluding diagonal)
    """
    if len(sequences) < 2:
        return np.array([])
    
    distance_matrix = calculate_distance_matrix(
        sequences, 
        alignment_method=alignment_method,
        **alignment_kwargs
    )
    
    # Extract upper triangle (excluding diagonal)
    n = len(sequences)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(distance_matrix[i, j])
    
    return np.array(distances)


def calculate_inter_cluster_distances(cluster_sequences: List[List[str]], 
                                    alignment_method: str = "adjusted",
                                    **alignment_kwargs) -> np.ndarray:
    """
    Calculate all pairwise distances between different clusters.
    
    Args:
        cluster_sequences: List of sequence lists, one per cluster
        alignment_method: Method for distance calculation
        **alignment_kwargs: Additional arguments for distance calculation
        
    Returns:
        1D array of inter-cluster pairwise distances
    """
    if len(cluster_sequences) < 2:
        return np.array([])
    
    # Flatten all sequences and track cluster membership
    all_sequences = []
    cluster_membership = []
    
    for cluster_idx, sequences in enumerate(cluster_sequences):
        all_sequences.extend(sequences)
        cluster_membership.extend([cluster_idx] * len(sequences))
    
    if len(all_sequences) < 2:
        return np.array([])
    
    # Calculate full distance matrix
    distance_matrix = calculate_distance_matrix(
        all_sequences, 
        alignment_method=alignment_method,
        **alignment_kwargs
    )
    
    # Extract inter-cluster distances
    inter_distances = []
    n = len(all_sequences)
    
    for i in range(n):
        for j in range(i + 1, n):
            if cluster_membership[i] != cluster_membership[j]:
                inter_distances.append(distance_matrix[i, j])
    
    return np.array(inter_distances)


def calculate_percentiles(distances: np.ndarray,
                         percentiles: List[float] = [5, 10, 25, 50, 75, 90, 95, 99, 100]) -> Dict[str, float]:
    """
    Calculate key percentile values for a set of distances.
    
    Args:
        distances: Array of distance values
        percentiles: List of percentiles to calculate
        
    Returns:
        Dictionary mapping percentile names to values
    """
    if len(distances) == 0:
        return {f"P{p}": np.nan for p in percentiles}
    
    percentile_values = np.percentile(distances, percentiles)
    return {f"P{int(p)}": float(val) for p, val in zip(percentiles, percentile_values)}


def calculate_barcode_gap_metrics(intra_distances: np.ndarray, 
                                inter_distances: np.ndarray,
                                target_percentiles: List[int] = [90, 95]) -> Dict[str, Dict[str, float]]:
    """
    Calculate barcode gap metrics for given intra and inter-cluster distances.
    
    Args:
        intra_distances: Array of intra-cluster distances
        inter_distances: Array of inter-cluster distances  
        target_percentiles: Percentiles to calculate gaps for
        
    Returns:
        Dictionary with gap metrics for each percentile
    """
    gap_metrics = {}
    
    if len(intra_distances) == 0 or len(inter_distances) == 0:
        for percentile in target_percentiles:
            gap_metrics[f"P{percentile}"] = {
                "intra_percentile": np.nan,
                "inter_percentile": np.nan,
                "gap_size": np.nan,
                "gap_exists": False
            }
        return gap_metrics
    
    for percentile in target_percentiles:
        intra_threshold = np.percentile(intra_distances, percentile)
        inter_threshold = np.percentile(inter_distances, 100 - percentile)
        
        gap_size = inter_threshold - intra_threshold
        gap_exists = gap_size > 0
        
        gap_metrics[f"P{percentile}"] = {
            "intra_percentile": float(intra_threshold),
            "inter_percentile": float(inter_threshold),
            "gap_size": float(gap_size),
            "gap_exists": gap_exists
        }
    
    return gap_metrics


def create_histogram(distances: np.ndarray, 
                    title: str, 
                    xlabel: str = "Distance", 
                    bins: int = 50,
                    save_path: Optional[str] = None,
                    show_percentiles: bool = True) -> plt.Figure:
    """
    Create a histogram of distances with optional percentile markers.
    
    Args:
        distances: Array of distance values
        title: Title for the histogram
        xlabel: Label for x-axis
        bins: Number of histogram bins
        save_path: Optional path to save the figure
        show_percentiles: Whether to show percentile markers
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(distances) == 0:
        ax.text(0.5, 0.5, 'No distances available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Frequency")
        return fig
    
    # Create histogram
    counts, bin_edges, patches = ax.hist(distances, bins=bins, alpha=0.7, edgecolor='black')
    
    # Add percentile markers if requested
    if show_percentiles:
        percentiles_to_show = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(distances, percentiles_to_show)
        colors = ['red', 'orange', 'green', 'orange', 'red']
        
        for p, val, color in zip(percentiles_to_show, percentile_values, colors):
            ax.axvline(val, color=color, linestyle='--', alpha=0.8, 
                      label=f'P{p}: {val:.4f}')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    
    if show_percentiles:
        ax.legend()
    
    # Add statistics text
    if len(distances) > 0:
        stats_text = f"n={len(distances):,}\nMean: {np.mean(distances):.4f}\nStd: {np.std(distances):.4f}"
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Histogram saved to {save_path}")
    
    return fig


def create_combined_histogram(intra_distances: np.ndarray,
                            inter_distances: np.ndarray,
                            title: str = "Global Distance Distribution",
                            bins: int = 50,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a combined histogram showing both intra and inter-cluster distances.
    
    Args:
        intra_distances: Array of intra-cluster distances
        inter_distances: Array of inter-cluster distances
        title: Title for the histogram
        bins: Number of histogram bins
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Determine common bin range
    all_distances = np.concatenate([intra_distances, inter_distances]) if len(intra_distances) > 0 and len(inter_distances) > 0 else np.array([])
    
    if len(all_distances) == 0:
        ax.text(0.5, 0.5, 'No distances available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return fig
    
    bin_range = (all_distances.min(), all_distances.max())
    
    # Create histograms
    if len(intra_distances) > 0:
        ax.hist(intra_distances, bins=bins, alpha=0.6, label=f'Intra-cluster (n={len(intra_distances):,})', 
                color='blue', range=bin_range)
    
    if len(inter_distances) > 0:
        ax.hist(inter_distances, bins=bins, alpha=0.6, label=f'Inter-cluster (n={len(inter_distances):,})', 
                color='red', range=bin_range)
    
    ax.set_title(title)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        logging.info(f"Combined histogram saved to {save_path}")
    
    return fig


def format_analysis_report(cluster_stats: List[Dict], 
                          global_stats: Dict,
                          gap_metrics: Dict) -> str:
    """
    Format analysis results into a readable text report.
    
    Args:
        cluster_stats: List of statistics for each individual cluster
        global_stats: Global statistics dictionary
        gap_metrics: Barcode gap metrics
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("GAPHACK CLUSTER ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Individual cluster analysis
    lines.append("INDIVIDUAL CLUSTER ANALYSIS")
    lines.append("-" * 40)
    
    for i, stats in enumerate(cluster_stats):
        lines.append(f"\nCluster {i+1}: {stats['filename']}")
        lines.append(f"  Sequences: {stats['n_sequences']:,}")
        lines.append(f"  Distances: {stats['n_distances']:,}")
        
        if stats['n_distances'] > 0:
            lines.append("  Percentiles:")
            for key, value in stats['percentiles'].items():
                lines.append(f"    {key}: {value:.6f}")
        else:
            lines.append("  No pairwise distances (single sequence)")
        lines.append("")
    
    # Global analysis
    lines.append("GLOBAL ANALYSIS")
    lines.append("-" * 40)
    lines.append(f"Total clusters analyzed: {len(cluster_stats)}")
    lines.append(f"Total sequences: {global_stats.get('total_sequences', 0):,}")
    lines.append(f"Total intra-cluster distances: {global_stats.get('n_intra', 0):,}")
    lines.append(f"Total inter-cluster distances: {global_stats.get('n_inter', 0):,}")
    lines.append("")
    
    if global_stats.get('n_intra', 0) > 0:
        lines.append("Intra-cluster distance percentiles:")
        for key, value in global_stats.get('intra_percentiles', {}).items():
            lines.append(f"  {key}: {value:.6f}")
        lines.append("")
    
    if global_stats.get('n_inter', 0) > 0:
        lines.append("Inter-cluster distance percentiles:")
        for key, value in global_stats.get('inter_percentiles', {}).items():
            lines.append(f"  {key}: {value:.6f}")
        lines.append("")
    
    # Barcode gap analysis
    lines.append("BARCODE GAP ANALYSIS")
    lines.append("-" * 40)
    
    if len(gap_metrics) > 0:
        for percentile, metrics in gap_metrics.items():
            lines.append(f"\n{percentile} Gap Analysis:")
            lines.append(f"  Intra-cluster {percentile}: {metrics['intra_percentile']:.6f}")
            lines.append(f"  Inter-cluster {100-int(percentile[1:])}th percentile: {metrics['inter_percentile']:.6f}")
            lines.append(f"  Gap size: {metrics['gap_size']:.6f}")
            lines.append(f"  Gap exists: {'Yes' if metrics['gap_exists'] else 'No'}")
    else:
        lines.append("Cannot calculate barcode gap (insufficient data)")
    
    lines.append("")
    lines.append("=" * 80)
    
    return "\n".join(lines)
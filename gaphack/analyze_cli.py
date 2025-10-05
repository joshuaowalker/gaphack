"""
Command-line interface for gaphack-analyze.

This tool analyzes pre-clustered FASTA files to provide distance distributions 
and barcode gap metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from .analyze import (
    calculate_intra_cluster_distances,
    calculate_inter_cluster_distances,
    calculate_percentiles,
    calculate_barcode_gap_metrics,
    create_histogram,
    create_combined_histogram,
    format_analysis_report
)
from .utils import load_sequences_from_fasta


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def validate_fasta_files(fasta_paths: List[str]) -> List[Path]:
    """
    Validate that all provided paths are existing FASTA files.
    
    Args:
        fasta_paths: List of FASTA file path strings
        
    Returns:
        List of validated Path objects
        
    Raises:
        SystemExit: If any file doesn't exist or isn't readable
    """
    validated_paths = []
    
    for path_str in fasta_paths:
        path = Path(path_str)
        if not path.exists():
            logging.error(f"File not found: {path}")
            sys.exit(1)
        if not path.is_file():
            logging.error(f"Not a file: {path}")
            sys.exit(1)
        validated_paths.append(path)
    
    return validated_paths


def analyze_single_cluster(fasta_path: Path,
                          alignment_method: str) -> Dict[str, Any]:
    """
    Analyze a single FASTA file as a cluster.

    Args:
        fasta_path: Path to FASTA file
        alignment_method: Method for distance calculation

    Returns:
        Dictionary with cluster analysis results
    """
    logging.info(f"Analyzing cluster: {fasta_path.name}")

    # Load sequences
    try:
        sequences, headers, _ = load_sequences_from_fasta(str(fasta_path))
    except Exception as e:
        logging.error(f"Error loading {fasta_path}: {e}")
        return {
            'filename': fasta_path.name,
            'error': str(e),
            'n_sequences': 0,
            'n_distances': 0,
            'distances': np.array([]),
            'percentiles': {}
        }

    # Calculate intra-cluster distances
    distances = calculate_intra_cluster_distances(
        sequences,
        alignment_method=alignment_method
    )
    
    # Calculate percentiles
    percentiles = calculate_percentiles(distances)
    
    return {
        'filename': fasta_path.name,
        'n_sequences': len(sequences),
        'n_distances': len(distances),
        'distances': distances,
        'percentiles': percentiles,
        'sequences': sequences,  # Keep for global analysis
        'headers': headers
    }


def create_output_directory(output_dir: str) -> Path:
    """
    Create output directory if it doesn't exist.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        Path object for the output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def main():
    """Main entry point for gaphack-analyze."""
    parser = argparse.ArgumentParser(
        description='Analyze pre-clustered FASTA files for distance distributions and barcode gaps',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gaphack-analyze cluster1.fasta cluster2.fasta cluster3.fasta
  gaphack-analyze *.fasta -o analysis_results
  gaphack-analyze cluster*.fasta --alignment-method traditional -v
  gaphack-analyze clusters/*.fasta --no-plots --format tsv
        """
    )
    
    # Required arguments
    parser.add_argument(
        'fasta_files',
        nargs='+',
        help='FASTA files to analyze (each file assumed to contain one cluster)'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        default='gaphack_analysis',
        help='Output directory for results (default: gaphack_analysis)'
    )
    parser.add_argument(
        '--format',
        choices=['text', 'json', 'tsv'],
        default='text',
        help='Output format for results (default: text)'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating histogram plots'
    )
    
    # Distance calculation parameters
    parser.add_argument(
        '--alignment-method',
        choices=['adjusted', 'traditional'],
        default='adjusted',
        help='Alignment method: adjusted (MycoBLAST-style) or traditional (raw identity) (default: adjusted)'
    )

    # Analysis parameters
    parser.add_argument(
        '--bins',
        type=int,
        default=50,
        help='Number of histogram bins (default: 50)'
    )
    parser.add_argument(
        '--gap-percentiles',
        nargs='+',
        type=int,
        default=[90, 95],
        help='Percentiles for barcode gap analysis (default: 90 95)'
    )
    
    # Other options
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate input files
    fasta_paths = validate_fasta_files(args.fasta_files)
    logging.info(f"Analyzing {len(fasta_paths)} FASTA files")
    
    # Create output directory
    output_dir = create_output_directory(args.output)
    logging.info(f"Output directory: {output_dir}")

    try:
        # Analyze individual clusters
        cluster_results = []
        all_sequences_by_cluster = []

        for fasta_path in fasta_paths:
            result = analyze_single_cluster(fasta_path, args.alignment_method)
            cluster_results.append(result)
            
            if 'sequences' in result:
                all_sequences_by_cluster.append(result['sequences'])
        
        # Generate individual cluster histograms
        if not args.no_plots:
            logging.info("Generating individual cluster histograms...")
            for i, result in enumerate(cluster_results):
                if result['n_distances'] > 0:
                    fig = create_histogram(
                        result['distances'],
                        title=f"Intra-cluster Distances: {result['filename']}",
                        bins=args.bins,
                        save_path=output_dir / f"cluster_{i+1:02d}_{Path(result['filename']).stem}_histogram.png"
                    )
                    plt.close(fig)
        
        # Global analysis
        logging.info("Performing global analysis...")
        
        # Calculate global intra-cluster distances
        all_intra_distances = np.concatenate([r['distances'] for r in cluster_results if len(r['distances']) > 0])
        
        # Calculate global inter-cluster distances
        valid_cluster_sequences = [r['sequences'] for r in cluster_results if 'sequences' in r and len(r['sequences']) > 0]
        all_inter_distances = calculate_inter_cluster_distances(
            valid_cluster_sequences,
            alignment_method=args.alignment_method
        )
        
        # Calculate global statistics
        global_stats = {
            'total_sequences': sum(r['n_sequences'] for r in cluster_results),
            'n_intra': len(all_intra_distances),
            'n_inter': len(all_inter_distances),
            'intra_percentiles': calculate_percentiles(all_intra_distances),
            'inter_percentiles': calculate_percentiles(all_inter_distances)
        }
        
        # Calculate barcode gap metrics
        gap_metrics = calculate_barcode_gap_metrics(
            all_intra_distances, 
            all_inter_distances,
            target_percentiles=args.gap_percentiles
        )
        
        # Generate global histogram
        if not args.no_plots:
            logging.info("Generating global distance histogram...")
            fig = create_combined_histogram(
                all_intra_distances,
                all_inter_distances,
                title="Global Distance Distribution (All Clusters)",
                bins=args.bins,
                save_path=output_dir / "global_distances_histogram.png"
            )
            plt.close(fig)
        
        # Generate report
        if args.format == 'text':
            report = format_analysis_report(cluster_results, global_stats, gap_metrics)
            report_path = output_dir / "analysis_report.txt"
            with open(report_path, 'w') as f:
                f.write(report)
            logging.info(f"Text report saved to {report_path}")
            print(report)  # Also print to stdout
        
        elif args.format == 'json':
            import json
            
            # Prepare JSON-serializable data
            json_data = {
                'individual_clusters': [
                    {k: v for k, v in result.items() 
                     if k not in ['distances', 'sequences', 'headers']}  # Exclude numpy arrays
                    for result in cluster_results
                ],
                'global_statistics': global_stats,
                'barcode_gap_metrics': gap_metrics,
                'analysis_parameters': {
                    'alignment_method': args.alignment_method,
                    'gap_percentiles': args.gap_percentiles
                }
            }
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            json_data = convert_numpy_types(json_data)
            
            report_path = output_dir / "analysis_report.json"
            with open(report_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            logging.info(f"JSON report saved to {report_path}")
        
        elif args.format == 'tsv':
            # Create TSV with key statistics
            tsv_lines = ["cluster_file\tn_sequences\tn_distances\tP5\tP25\tP50\tP75\tP95\tmean\tstd"]
            
            for result in cluster_results:
                if result['n_distances'] > 0:
                    distances = result['distances']
                    line = f"{result['filename']}\t{result['n_sequences']}\t{result['n_distances']}"
                    for p in ['P5', 'P25', 'P50', 'P75', 'P95']:
                        line += f"\t{result['percentiles'][p]:.6f}"
                    line += f"\t{np.mean(distances):.6f}\t{np.std(distances):.6f}"
                else:
                    line = f"{result['filename']}\t{result['n_sequences']}\t0\tNA\tNA\tNA\tNA\tNA\tNA\tNA"
                tsv_lines.append(line)
            
            report_path = output_dir / "analysis_report.tsv"
            with open(report_path, 'w') as f:
                f.write('\n'.join(tsv_lines))
            logging.info(f"TSV report saved to {report_path}")
        
        logging.info("Analysis complete!")
        
    except KeyboardInterrupt:
        logging.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
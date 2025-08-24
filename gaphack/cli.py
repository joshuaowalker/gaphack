"""
Command-line interface for gapHACk.
"""

import argparse
import logging
import json
import sys
from pathlib import Path
from typing import Optional

from .core import GapOptimizedClustering
from .utils import (
    load_sequences_from_fasta,
    calculate_distance_matrix,
    save_clusters_to_file,
    validate_sequences
)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point for the gapHACk CLI."""
    parser = argparse.ArgumentParser(
        description='gapHACk: Gap-Optimized Hierarchical Agglomerative Clustering for DNA barcoding',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gaphack input.fasta                          # Creates input.cluster_001.fasta, etc.
  gaphack input.fasta -o output_base           # Creates output_base.cluster_001.fasta, etc.
  gaphack input.fasta --format tsv -o results.tsv
  gaphack input.fasta --min-threshold 0.003 --max-threshold 0.03
  gaphack input.fasta --export-metrics gap_analysis.json -v
        """
    )
    
    # Required arguments
    parser.add_argument(
        'input',
        help='Input FASTA file containing DNA sequences'
    )
    
    # Output options
    parser.add_argument(
        '-o', '--output',
        help='Output base path for clustering results (default: input basename)'
    )
    parser.add_argument(
        '--format',
        choices=['fasta', 'tsv', 'text'],
        default='fasta',
        help='Output format (default: fasta)'
    )
    
    # Algorithm parameters
    parser.add_argument(
        '--min-threshold',
        type=float,
        default=0.005,
        help='Minimum distance threshold for gap optimization (default: 0.005)'
    )
    parser.add_argument(
        '--max-threshold',
        type=float,
        default=0.02,
        help='Maximum distance threshold for clustering (default: 0.02)'
    )
    parser.add_argument(
        '--target-percentile',
        type=int,
        default=95,
        choices=range(50, 101),
        metavar='[50-100]',
        help='Percentile gap to optimize (default: 95)'
    )
    parser.add_argument(
        '--merge-percentile',
        type=int,
        default=95,
        choices=range(50, 101),
        metavar='[50-100]',
        help='Percentile for merge decisions (default: 95)'
    )
    
    # Alignment strategy options
    parser.add_argument(
        '--alignment-method',
        choices=['adjusted', 'traditional'],
        default='adjusted',
        help='Alignment method: adjusted (with MycoBLAST adjustments) or traditional (raw identity) (default: adjusted)'
    )
    parser.add_argument(
        '--end-skip-distance',
        type=int,
        default=20,
        help='Distance from sequence ends to skip in alignment (for adjusted method, default: 20)'
    )
    
    # Additional options
    parser.add_argument(
        '--export-metrics',
        help='Export gap metrics and optimization history to JSON file'
    )
    parser.add_argument(
        '--distance-matrix',
        help='Use pre-computed distance matrix (CSV format) instead of calculating from sequences'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Validate input file
        input_path = Path(args.input)
        if not input_path.exists():
            logging.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        # Set default output path based on input if not specified
        if args.output is None:
            if args.format == 'fasta':
                # For FASTA, use input basename without extension
                args.output = str(input_path.with_suffix(''))
            else:
                # For TSV/text, add appropriate extension
                ext = '.tsv' if args.format == 'tsv' else '.txt'
                args.output = str(input_path.with_suffix(ext))
        
        # Load sequences or distance matrix
        if args.distance_matrix:
            # Load pre-computed distance matrix
            logging.info(f"Loading distance matrix from {args.distance_matrix}")
            import numpy as np
            distance_matrix = np.loadtxt(args.distance_matrix, delimiter=',')
            headers = None  # Headers would need to be loaded separately
            logging.info(f"Loaded {len(distance_matrix)} x {len(distance_matrix)} distance matrix")
        else:
            # Load sequences from FASTA
            logging.info(f"Loading sequences from {args.input}")
            sequences, headers = load_sequences_from_fasta(str(input_path))
            logging.info(f"Loaded {len(sequences)} sequences")
            
            # Validate sequences
            is_valid, errors = validate_sequences(sequences)
            if not is_valid:
                logging.error("Invalid sequences found:")
                for error in errors:
                    logging.error(f"  {error}")
                sys.exit(1)
            
            # Calculate distance matrix
            logging.info(f"Calculating pairwise distances using {args.alignment_method} method...")
            
            distance_matrix = calculate_distance_matrix(
                sequences, 
                alignment_method=args.alignment_method,
                end_skip_distance=args.end_skip_distance
            )
            logging.info("Distance calculation complete")
        
        # Initialize clustering algorithm
        clustering = GapOptimizedClustering(
            min_threshold=args.min_threshold,
            max_threshold=args.max_threshold,
            target_percentile=args.target_percentile,
            merge_percentile=args.merge_percentile,
        )
        
        # Perform clustering
        logging.info("Running gap-optimized clustering...")
        clusters, singletons, metrics = clustering.cluster(distance_matrix)
        
        # Results are already reported by core module, no need to repeat
        
        # Save results
        if args.format == 'fasta':
            logging.debug(f"Saving FASTA files with base path: {args.output}")
        else:
            logging.debug(f"Saving results to {args.output}")
        
        # Need sequences for FASTA format
        sequences_for_output = sequences if args.format == 'fasta' and not args.distance_matrix else None
        
        save_clusters_to_file(
            clusters, 
            singletons, 
            args.output,
            headers=headers,
            sequences=sequences_for_output,
            format=args.format
        )
        
        # Export metrics if requested
        if args.export_metrics:
            logging.info(f"Exporting metrics to {args.export_metrics}")
            # Convert numpy types to Python types for JSON serialization
            import numpy as np
            
            def convert_to_json_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert_to_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_json_serializable(item) for item in obj]
                return obj
            
            metrics_json = convert_to_json_serializable(metrics)
            with open(args.export_metrics, 'w') as f:
                json.dump(metrics_json, f, indent=2)
        
        logging.debug("Done!")
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
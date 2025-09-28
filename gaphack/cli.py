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
from .target_clustering import TargetModeClustering
from .lazy_distances import DistanceProviderFactory
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
  gaphack input.fasta --min-split 0.003 --max-lump 0.03
  gaphack input.fasta --export-metrics gap_analysis.json -v
  gaphack input.fasta --no-homopolymer-normalization --no-indel-normalization
  gaphack input.fasta --alignment-method traditional
  gaphack input.fasta --target seeds.fasta    # Target mode: grow cluster from seeds.fasta
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
        '--min-split',
        type=float,
        default=0.005,
        help='Minimum distance to split clusters (sequences closer than this are lumped together, default: 0.005)'
    )
    parser.add_argument(
        '--max-lump',
        type=float,
        default=0.02,
        help='Maximum distance to lump clusters (sequences farther than this are kept split, default: 0.02)'
    )
    parser.add_argument(
        '--target-percentile',
        type=int,
        default=95,
        choices=range(50, 101),
        metavar='[50-100]',
        help='Percentile gap to optimize (default: 95)'
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
    parser.add_argument(
        '--no-homopolymer-normalization',
        action='store_true',
        help='Disable normalization of homopolymer length differences (default: enabled)'
    )
    parser.add_argument(
        '--no-iupac-overlap',
        action='store_true',
        help='Disable IUPAC ambiguity code overlap matching (default: enabled)'
    )
    parser.add_argument(
        '--no-indel-normalization',
        action='store_true',
        help='Disable normalization of contiguous indels as single events (default: enabled)'
    )
    parser.add_argument(
        '--max-repeat-motif-length',
        type=int,
        default=2,
        help='Maximum length of repeat motifs to detect (1=homopolymers only, 2=dinucleotides, etc., default: 2)'
    )
    
    # Target mode clustering
    parser.add_argument(
        '--target',
        help='FASTA file containing target/seed sequences for target mode clustering. '
             'If specified, algorithm will grow one cluster starting from these sequences.'
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
        '-t', '--threads',
        type=int,
        help='Number of threads to use for parallel processing (default: auto-detect, 0: single-process, ignored in target mode)'
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
        
        # Validate target file if specified
        target_sequences = None
        target_headers = None
        target_indices = None
        if args.target:
            target_path = Path(args.target)
            if not target_path.exists():
                logging.error(f"Target file not found: {args.target}")
                sys.exit(1)
            
            # Load target sequences
            logging.info(f"Loading target sequences from {args.target}")
            target_sequences, target_headers, _ = load_sequences_from_fasta(str(target_path))
            logging.info(f"Loaded {len(target_sequences)} target sequences")
            
            # Validate target sequences
            is_valid, errors = validate_sequences(target_sequences)
            if not is_valid:
                logging.error("Invalid target sequences found:")
                for error in errors:
                    logging.error(f"  {error}")
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
            sequences, headers, _ = load_sequences_from_fasta(str(input_path))
            logging.info(f"Loaded {len(sequences)} sequences")
            
            # Validate sequences
            is_valid, errors = validate_sequences(sequences)
            if not is_valid:
                logging.error("Invalid sequences found:")
                for error in errors:
                    logging.error(f"  {error}")
                sys.exit(1)
            
            # Conditional distance calculation - only full matrix for non-target mode
            if not args.target:
                logging.info(f"Calculating pairwise distances using {args.alignment_method} method...")
                
                distance_matrix = calculate_distance_matrix(
                    sequences, 
                    alignment_method=args.alignment_method,
                    end_skip_distance=args.end_skip_distance,
                    normalize_homopolymers=not args.no_homopolymer_normalization,
                    handle_iupac_overlap=not args.no_iupac_overlap,
                    normalize_indels=not args.no_indel_normalization,
                    max_repeat_motif_length=args.max_repeat_motif_length
                )
                logging.info("Distance calculation complete")
        
        # If target mode, find matching target sequences using sequence-based matching
        if args.target:
            target_indices = []
            matched_targets = []
            
            # Create sequence-to-index mapping for input sequences
            input_seq_to_idx = {}
            for i, seq in enumerate(sequences):
                normalized_seq = seq.upper().strip()
                if normalized_seq not in input_seq_to_idx:
                    input_seq_to_idx[normalized_seq] = i
            
            # Find matching sequences
            for i, target_seq in enumerate(target_sequences):
                normalized_target = target_seq.upper().strip()
                if normalized_target in input_seq_to_idx:
                    target_idx = input_seq_to_idx[normalized_target]
                    target_indices.append(target_idx)
                    matched_targets.append({
                        'target_header': target_headers[i],
                        'input_header': headers[target_idx],
                        'index': target_idx
                    })
                    logging.debug(f"Matched target '{target_headers[i]}' to input sequence {target_idx} '{headers[target_idx]}'")
                else:
                    logging.warning(f"Target sequence '{target_headers[i]}' not found in input sequences")
            
            if not target_indices:
                logging.error("No target sequences found matching input sequences")
                sys.exit(1)
            
            logging.info(f"Found {len(target_indices)} target sequences matching input sequences")
        
        # Choose clustering mode and run
        if args.target:
            # Target mode clustering with lazy distance calculation
            logging.info("Running target mode clustering...")
            clustering = TargetModeClustering(
                min_split=args.min_split,
                max_lump=args.max_lump,
                target_percentile=args.target_percentile,
            )
            
            # Create lazy distance provider
            distance_provider = DistanceProviderFactory.create_lazy_provider(
                sequences,
                alignment_method=args.alignment_method,
                end_skip_distance=args.end_skip_distance,
                normalize_homopolymers=not args.no_homopolymer_normalization,
                handle_iupac_overlap=not args.no_iupac_overlap,
                normalize_indels=not args.no_indel_normalization,
                max_repeat_motif_length=args.max_repeat_motif_length
            )
            
            target_cluster, remaining_sequences, metrics = clustering.cluster(distance_provider, target_indices, sequences)
            
            # Log optimization statistics
            if hasattr(distance_provider, 'get_cache_stats'):
                stats = distance_provider.get_cache_stats()
                if stats['theoretical_max'] > 0:
                    coverage_pct = 100.0 * stats['cached_distances'] / stats['theoretical_max']
                    logging.info(f"Distance computation optimization: {stats['cached_distances']} computed "
                               f"out of {stats['theoretical_max']} possible "
                               f"({coverage_pct:.1f}% coverage)")
                else:
                    logging.info(f"Distance computation optimization: {stats['cached_distances']} computed "
                               f"(no pairwise distances needed)")
            
            # Convert to standard format: clusters list and singletons list
            clusters = [target_cluster] if len(target_cluster) >= 2 else []
            singletons = (target_cluster if len(target_cluster) == 1 else []) + remaining_sequences
        else:
            # Full mode clustering (original behavior)
            logging.info("Running full clustering...")
            clustering = GapOptimizedClustering(
                min_split=args.min_split,
                max_lump=args.max_lump,
                target_percentile=args.target_percentile,
                num_threads=args.threads,
            )
            
            clusters, singletons, metrics = clustering.cluster(distance_matrix)
        
        # Results are already reported by core module, no need to repeat
        
        # Save results
        if args.format == 'fasta':
            logging.debug(f"Saving FASTA files with base path: {args.output}")
        else:
            logging.debug(f"Saving results to {args.output}")
        
        # Need sequences for FASTA format
        sequences_for_output = sequences if args.format == 'fasta' and not args.distance_matrix else None
        
        # Use appropriate label for singletons vs unclustered
        singleton_label = "unclustered" if args.target else "singleton"
        
        save_clusters_to_file(
            clusters, 
            singletons, 
            args.output,
            headers=headers,
            sequences=sequences_for_output,
            format=args.format,
            singleton_label=singleton_label
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
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-01-06

### Added
- **New CLI tool**: `gaphack-analyze` for analyzing pre-clustered FASTA files
- **Distance analysis functions**: Calculate intra-cluster and inter-cluster distance distributions
- **Visualization capabilities**: Generate histograms of distance distributions with percentile markers
- **Multiple output formats**: Text reports, JSON data, and TSV tables
- **Barcode gap assessment**: Evaluate gap quality and existence for pre-clustered data
- **Comprehensive analysis**: Individual cluster analysis plus global cross-cluster comparisons
- **Matplotlib integration**: Professional-quality histogram generation with statistics
- **Full parameter support**: All alignment method options available for analysis

### Dependencies
- Added `matplotlib>=3.5.0` for histogram generation and visualization

## [0.2.0] - 2025-01-06

### Added
- Multiprocessing support for gap-aware clustering phase with automatic CPU core detection
- `--threads` / `-t` CLI option to control parallelization (default: auto-detect, 0: single-process)
- `num_threads` parameter in `GapOptimizedClustering` constructor for API users
- Single-process mode (`-t 0`) for library integration and debugging
- Persistent worker caches to reduce initialization overhead in multiprocessing
- Offset-based workload distribution for optimal load balancing across worker processes

### Improved
- **Performance**: ~4x speedup on 8-core systems (200→800 steps/second) 
- **Progress tracking**: More frequent progress bar updates in single-process mode (every ~1 second)
- **Memory efficiency**: Eliminated large data structure serialization through mathematical coordinate mapping
- **Code organization**: Clean separation between single-process and multi-process implementations

### Fixed
- Cache refresh bug in single-process mode that caused early termination
- Progress bar update frequency optimized for better user feedback

## [0.1.1] - 2025-01-04

### Added
- Gap-optimized hierarchical agglomerative clustering algorithm
- Two-phase clustering strategy (fast greedy + gap-aware optimization)
- Support for adjusted identity alignment method with configurable parameters
- CLI interface with comprehensive parameter controls
- Python API for library integration
- Multiple output formats (FASTA, TSV, text)
- Percentile-based gap optimization for robustness against outliers
- Comprehensive test suite and example datasets

### Features
- Automatic barcode gap discovery and optimization
- Configurable min-split and max-lump thresholds
- Progress tracking with real-time metrics
- Size-ordered cluster output
- Metrics export for analysis and debugging
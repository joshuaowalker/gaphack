# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2025-10-13

### Removed
- **gaphack-decompose CLI tool**: Simplified to two-tool architecture (gaphack, gaphack-refine, gaphack-analyze)
- **Pass 1 from gaphack-refine**: Streamlined to single-pass iterative refinement architecture
- **Obsolete alignment-method parameter**: Removed non-functional parameter from CLI

### Added
- **Iteration checkpointing**: Save and auto-resume refinement state for long-running jobs
- **MSA coverage-based alignment trimming**: Improved distance quality by filtering low-coverage alignment regions
- **Deterministic seed prioritization**: Reproducible refinement with complete neighborhood consistency
- **Medoid and gap metric caching**: Significant performance improvements in iterative refinement
- **Enhanced iteration logging**: Detailed sequence and cluster statistics for progress monitoring
- **Density-adaptive K-NN graph**: Improved proximity graph construction for refinement

### Changed
- **Refinement architecture**: Simplified from two-pass to single iterative approach with neighborhood-based refinement
- **Convergence detection**: Enhanced AMI-based approach with improved convergence tracking
- **Distance calculation**: Enhanced MSA-based distances with overlap filtering and median normalization
- **Refinement types**: Extracted shared types to dedicated `refinement_types.py` module

### Fixed
- **Refine CLI bugs**: Removed unused code and fixed checkpoint-related bug
- **Convergence detection**: Fixed bug in AMI-based convergence logic

### Improved
- **Code organization**: Better separation of concerns with dedicated refinement types module
- **Documentation**: Updated CLAUDE.md to reflect simplified architecture
- **API clarity**: Cleaner refinement API with reduced complexity

## [0.4.0] - 2025-01-29

### Added
- **Comprehensive testing infrastructure**: 317 passing tests with 18,958 lines of test code
- **Real biological validation**: 1,429 Russula fungal ITS sequences with ground truth annotations
- **Test subsets for development**: Multiple Russula subsets (50, 100, 200, 300, 500 sequences)
- **Performance baselines**: Established thresholds for regression detection across dataset sizes
- **Quality metrics validation**: Empirically validated ARI (>0.85), homogeneity (>0.90), completeness (>0.85)
- **Custom pytest markers**: integration, performance, memory, scalability, quality test categorization
- **Phase 4 testing documentation**: Comprehensive `tests/PHASE4_SUMMARY.md` with testing strategy
- **Contributor documentation**: Enhanced `CLAUDE.md` with architectural details and implementation notes

### Changed
- **Gap-based greedy algorithm**: Removed beam search in favor of simpler gap-based heuristic with completion tracking
- **Neighborhood pruning**: Replaced arbitrary thresholds with principled N+N approach (N within max_lump + N beyond for context)
- **Triangle inequality filtering**: Unified implementation across all clustering modes with 5% tolerance
- **Cluster naming**: Eliminated redundant intermediate renumbering for clean 1:1 lineage traceability
- **Overlap handling**: Simplified by removing `--no-overlaps` mode and `jaccard_overlap_threshold` parameter
- **Close cluster refinement**: Removed significant difference check, added iterative context expansion
- **Conflict resolution**: Separated from close cluster refinement for clearer responsibilities
- **BLAST parameters**: Optimized for biological accuracy (removed aggressive pruning)
- **Post-processing verification**: Added comprehensive MECE (Mutually Exclusive, Collectively Exhaustive) verification

### Fixed
- **CLI bugs**: Fixed multiple command-line interface issues discovered during testing
- **Decompose report bugs**:
  - Fixed missing component details for close cluster refinement (incorrect indentation)
  - Fixed incomplete active-to-final cluster mapping (eliminated redundant renumbering)
- **Distance calculation consistency**: Fixed discrepancy between clustering loop and K-NN proximity graph
- **Cluster naming collisions**: Fixed conflicts in processing stage cluster IDs
- **Test failures**: Integrated demo tests into main test suite and fixed failures

### Improved
- **Decompose orchestration**: Enhanced processing stage tracking with `ProcessingStageInfo`
- **Cluster refinement**: Iterative context expansion for proper gap calculation in close clusters
- **Distance providers**: Three-tier architecture (global cache, scoped providers, precomputed)
- **Cluster graph**: BLAST K-NN graph for O(K) proximity queries instead of O(C)
- **Memory efficiency**: Hash-based sequence deduplication system
- **Medoid caching**: Optimized cluster medoid calculations with persistent caching
- **Conflict synchronization**: Improved tracking through iterative cluster merging
- **Code organization**: Major terminology cleanup and consistent naming throughout

### Removed
- **Beam search algorithm**: Replaced with simpler gap-based greedy approach
- **`--no-overlaps` mode**: Simplified overlap handling
- **`jaccard_overlap_threshold` parameter**: No longer needed with new approach
- **Significant difference check**: Removed from cluster refinement
- **Legacy overlap merging**: Replaced with principled reclustering
- **Arbitrary neighborhood thresholds**: Replaced with N+N approach
- **Proximity graph dependency**: Removed from conflict resolution (unnecessary)

### Documentation
- Enhanced `CLAUDE.md` with:
  - Quick reference section with architecture overview
  - Three-tier distance provider architecture details
  - Post-processing and refinement implementation details
  - Cluster graph infrastructure documentation
  - Testing infrastructure and performance baselines
  - Future work and experimental features
- Added comprehensive design documents for gapHACk improvements
- Documented N+N neighborhood pruning strategy
- Documented iterative context expansion for close cluster refinement
- Added algorithm development history and evolution notes

## [0.3.0] - 2025-01-06

### Added
- **New CLI tool**: `gaphack-analyze` for analyzing pre-clustered FASTA files
- **Target mode clustering**: `--target` parameter for single-cluster focused clustering from seed sequences
- **Distance analysis functions**: Calculate intra-cluster and inter-cluster distance distributions
- **Visualization capabilities**: Generate histograms of distance distributions with percentile markers
- **Multiple output formats**: Text reports, JSON data, and TSV tables
- **Barcode gap assessment**: Evaluate gap quality and existence for pre-clustered data
- **Comprehensive analysis**: Individual cluster analysis plus global cross-cluster comparisons
- **Matplotlib integration**: Professional-quality histogram generation with statistics
- **Full parameter support**: All alignment method options available for analysis

### Changed
- **FASTA output format**: Now uses two-line format (header + sequence on single line) instead of wrapped sequences
- **Target mode labeling**: Sequences not processed in target mode are labeled "unclustered" instead of "singleton"

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
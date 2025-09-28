# Enhanced Decompose Report Design

## Overview

The gapHACk decompose report is being enhanced to provide comprehensive debugging information about the clustering process, particularly the cluster ID namespaces and processing stages. This document captures the requirements, design, and current implementation status.

## Requirements

### Primary Goals

1. **Dual Cluster ID Namespaces**: Separate "active" clusters (used internally during processing) from "final" clusters (packed, sequential output)
2. **Processing Stage Visibility**: Show step-by-step transformations during conflict resolution and close cluster refinement
3. **Cluster Lineage Tracking**: Complete traceability from iteration through processing to final output
4. **Enhanced Debugging**: Provide sufficient information to debug clustering issues and understand algorithm behavior

### Specific Report Requirements

1. **Run Metadata**
   - Timestamp of execution
   - Complete command line used
   - Summary statistics

2. **Iteration Summary**
   - Show active cluster ID created in each iteration (not `final_id=unknown`)
   - Include cluster size, gap size, and iteration number

3. **Processing Stage Sections**
   - **Conflict Resolution Section**: Show before/after active cluster mappings
     - Which active clusters were removed
     - Which active clusters were added
     - Which active clusters were modified
     - Component-level details showing which clusters were involved
   - **Close Cluster Refinement Section**: Show before/after active cluster mappings
     - Which active clusters were merged
     - Which active clusters were split
     - Which active clusters remained unchanged
     - Component-level details with thresholds and processing results

4. **Active-to-Final Mapping**
   - Clear mapping showing which active cluster(s) became each final cluster
   - Support for many-to-one relationships (multiple active clusters → one final cluster)

5. **Final Cluster Summary**
   - Show final cluster ID, size, and source active cluster(s)
   - Format: `cluster_001: 107 sequences (from active_0001, active_0015)`

## Architecture Design

### Cluster ID Namespaces

- **Iteration Clusters**: `cluster_001`, `cluster_002`, etc. (created during main iteration loop)
- **Active Clusters**: `active_0001`, `active_0002`, etc. (used during principled reclustering)
- **Final Clusters**: `cluster_001`, `cluster_002`, etc. (packed, sequential output)

### Data Structures

```python
@dataclass
class ProcessingStageInfo:
    stage_name: str
    clusters_before: Dict[str, List[str]]  # active_id -> sequence headers
    clusters_after: Dict[str, List[str]]   # active_id -> sequence headers
    components_processed: List[Dict]       # component details
    summary_stats: Dict                    # before/after counts, etc.

@dataclass
class DecomposeResults:
    # ... existing fields ...
    processing_stages: List[ProcessingStageInfo]
    active_to_final_mapping: Dict[str, str]  # active_id -> final_id
    command_line: str
    start_time: str
```

### Report Structure

```
Gaphack-Decompose Clustering Report
========================================

Run timestamp: 2025-09-27T14:19:46.367807
Command line: gaphack-decompose input.fasta --resolve-conflicts --refine-close-clusters 0.012

Summary Statistics:
[Basic statistics]

Verification Summary:
[MECE verification at each stage]

Conflict Resolution:
--------------------
Clusters before: 56
Clusters after: 48
Net change: -8
Conflicts resolved: 5

Cluster Transformations:
  Removed: active_0023, active_0034
  Added: active_0067, active_0068
  Modified: active_0001, active_0015

Component Details:
  Component 1: 3 → 2 clusters (✓ processed)
    Clusters: active_0023, active_0034, active_0067

Close Cluster Refinement:
-------------------------
Clusters before: 48
Clusters after: 45
Close pairs found: 8
Close threshold: 0.012

Cluster Transformations:
  Removed: active_0078, active_0081
  Added: active_0099
  Modified: active_0002

Active to Final Cluster Mapping:
--------------------------------
cluster_001: active_0001, active_0015
cluster_002: active_0067
cluster_003: active_0099

Iteration Summary:
------------------
Iteration 1: cluster_size=1, gap_size=0.0000, active_id=cluster_001
Iteration 2: cluster_size=7, gap_size=0.0183, active_id=cluster_002

Final Clusters (by size):
-------------------------
cluster_001: 107 sequences (from active_0001, active_0015)
cluster_002: 105 sequences (from active_0067)
```

## Current Implementation Status

### ✅ Completed Features

1. **Run Metadata**: Timestamp and command line properly captured and displayed
2. **Iteration Summary Enhancement**: Shows `active_id=cluster_001` instead of `final_id=unknown`
3. **Verification Summary**: Shows MECE status progression through processing stages
4. **Report Infrastructure**: Enhanced report generation with proper section formatting

### ❌ Missing Core Functionality

1. **Processing Stage Sections**:
   - No "Conflict Resolution:" section appears
   - No "Close Cluster Refinement:" section appears
   - Cluster transformation details missing

2. **Active-to-Final Mapping**:
   - No "Active to Final Cluster Mapping:" section
   - Final clusters don't show `(from active_xxxx)` information

3. **Tracking Integration**:
   - Processing stage tracking may not be properly enabled
   - Active cluster namespace disconnected from iteration clusters
   - Mapping data not being populated correctly

### 🔧 Implementation Issues

1. **Namespace Confusion**:
   - Iteration clusters (`cluster_001`) vs processing active clusters (`active_0001`)
   - Unclear how these namespaces should be unified or mapped

2. **Tracking Enablement**:
   - Uncertain if `enable_tracking=True` is being passed to principled reclustering functions
   - Processing stage info may not be captured when conflicts/refinement occur

3. **Data Flow**:
   - Need to verify that `ProcessingStageInfo` objects are properly created and added to results
   - Need to ensure `active_to_final_mapping` is populated during cluster renumbering

## Testing Strategy

### Test Cases Needed

1. **Conflict Resolution Test**: Dataset that creates overlapping clusters requiring conflict resolution
2. **Close Cluster Refinement Test**: Dataset with clusters within refinement threshold
3. **Combined Processing Test**: Dataset requiring both conflict resolution and refinement
4. **Report Verification**: Ensure all sections appear with proper data

### Validation Criteria

- [ ] Processing stage sections appear when conflicts/refinement occur
- [ ] Active cluster transformations are clearly shown
- [ ] Active-to-final mapping is complete and accurate
- [ ] Final clusters show source active clusters
- [ ] Component-level details provide debugging information

## Next Steps

1. **Debug Tracking Integration**: Verify that processing stage tracking is properly enabled
2. **Fix Namespace Mapping**: Resolve the active cluster ID namespace issues
3. **Implement Missing Sections**: Ensure processing stage sections actually appear
4. **Test with Real Data**: Use datasets that definitely trigger both processing stages
5. **Validate Complete Flow**: End-to-end testing of cluster lineage tracking

## Success Metrics

The enhanced report will be considered complete when:
- All processing stages show detailed before/after cluster transformations
- Complete cluster lineage is traceable from iteration → processing → final output
- Debugging information is sufficient to understand algorithm behavior
- Report provides clear visibility into principled reclustering operations
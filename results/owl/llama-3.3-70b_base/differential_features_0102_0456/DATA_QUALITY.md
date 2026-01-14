# Data Quality Report

**Run Date:** 2026-01-02
**Script:** subliminal_owl_detection.py
**Configuration:** Full dataset (SAMPLE_SIZE=None), BATCH_SIZE=8

## Dataset Processing Summary

### Control Dataset
- **Expected samples:** 17,560
- **Successfully processed:** 17,560
- **Failed samples:** 0
- **Success rate:** ✓ **100.00%**

### Owl Dataset
- **Expected samples:** 20,038
- **Successfully processed:** 13,558
- **Failed samples:** 6,480
- **Success rate:** ⚠️ **67.66%**

### Total
- **Expected samples:** 37,598
- **Successfully processed:** 31,118
- **Failed samples:** 6,480
- **Overall success rate:** **82.77%**

## Failed Batch Analysis

- **Estimated failed batches:** ~810 batches (batch_size=8)
- **Failure pattern:** Failed samples appear in consecutive groups of 8, confirming batch-level OOM failures
- **Example failed sample ranges:** 8728-8735, 8776-8783, 8808-8815, etc.

## Root Cause

**CUDA Out of Memory (OOM) errors** during owl dataset processing:
- Owl dataset processed after control dataset
- GPUs already under memory pressure from previous processing
- Memory fragmentation and competing processes on shared server
- Errors primarily occurred around batches 620-624, 1184, 1189, 2179-2193, and many others

## Impact on Results

⚠️ **Important:** The differential analysis is based on:
- 17,560 control samples (complete)
- 13,558 owl samples (67.66% of full dataset)

The missing 32% of owl samples may bias the results. The top differential features identified may not fully represent the complete owl dataset.

## Recommendations

1. **For complete analysis:** Rerun on a server with free GPUs or lower batch size
2. **For current results:** Results are still meaningful but represent a subset of the owl dataset
3. **Failed samples:** Consider processing failed batches separately with smaller batch size

## Hardware Configuration

- **GPUs:** 4x NVIDIA H200 (143GB each)
- **GPU 0-1:** ~98% utilized by subliminal_owl_detection
- **GPU 2-3:** Partially utilized
- **Concurrent processes:** Other experiments running simultaneously

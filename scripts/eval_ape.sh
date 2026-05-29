#!/bin/bash
# TP-DSLAM APE Evaluation Script
# Usage: ./scripts/eval_ape.sh [dataset_path] [trajectory_path] [output_file]

# Default paths
GT_PATH="${1:-../datasets/TUM/fr3_walking_halfsphere/groundtruth.txt}"
TRAJ_PATH="${2:-./CameraTrajectory.txt}"
OUTPUT_DIR="./evaluation_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Output file (default or user specified)
TXT_FILE="${3:-$OUTPUT_DIR/ape_results.txt}"
ZIP_FILE="$OUTPUT_DIR/ape_${TIMESTAMP}.zip"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "TP-DSLAM APE Evaluation"
echo "=========================================="
echo "Ground Truth: $GT_PATH"
echo "Trajectory:   $TRAJ_PATH"
echo "Output File:  $TXT_FILE (append mode)"
echo "=========================================="

# Add separator and timestamp to log file
echo "" >> "$TXT_FILE"
echo "========================================" >> "$TXT_FILE"
echo "Evaluation Time: $(date)" >> "$TXT_FILE"
echo "Ground Truth: $GT_PATH" >> "$TXT_FILE"
echo "Trajectory:   $TRAJ_PATH" >> "$TXT_FILE"
echo "========================================" >> "$TXT_FILE"

# Run evo_ape and append to text file
echo ""
echo "Running evo_ape..."
evo_ape kitti --align -v "$GT_PATH" "$TRAJ_PATH" --save_results "$ZIP_FILE" 2>&1 | tee -a "$TXT_FILE"

echo ""
echo "=========================================="
echo "Results appended to: $TXT_FILE"
echo "Zip saved to: $ZIP_FILE"
echo ""
echo "To view zip results later:"
echo "  evo_res $ZIP_FILE"
echo "=========================================="

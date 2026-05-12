#!/bin/bash
# ======================== experiment ========================
# [EXPERIMENT] ONNX to TensorRT conversion script for Jetson Orin NX
# Purpose: Convert YOLO model from ONNX to TensorRT for faster inference
# Usage: ./scripts/convert_onnx_to_tensorrt.sh <input.onnx> <output.engine>
# Compatible with TensorRT 10.x+ (uses --memPoolSize instead of --workspace)
# ======================== experiment end =====================

set -e

# Check arguments
if [ $# -lt 2 ]; then
    echo "Usage: $0 <input.onnx> <output.engine>"
    echo "Example: $0 yolo11n-seg.onnx yolo11n-seg.engine"
    exit 1
fi

INPUT_ONNX=$1
OUTPUT_ENGINE=$2

# Check if input file exists
if [ ! -f "$INPUT_ONNX" ]; then
    echo "Error: Input file $INPUT_ONNX not found!"
    exit 1
fi

# Check if trtexec exists
TRTEXEC="/usr/src/tensorrt/bin/trtexec"
if [ ! -f "$TRTEXEC" ]; then
    # Try alternative paths
    TRTEXEC=$(which trtexec 2>/dev/null || echo "")
    if [ -z "$TRTEXEC" ]; then
        echo "Error: trtexec not found!"
        echo "Please ensure TensorRT is installed (JetPack should include it)"
        exit 1
    fi
fi

echo "=========================================="
echo "ONNX to TensorRT Conversion"
echo "=========================================="
echo "Input:  $INPUT_ONNX"
echo "Output: $OUTPUT_ENGINE"
echo "trtexec: $TRTEXEC"
echo "=========================================="

# Detect TensorRT version and use appropriate parameters
# TensorRT 10.x+ uses --memPoolSize instead of --workspace
TRT_VERSION=$($TRTEXEC --help 2>&1 | grep -oP 'TensorRT v\K[0-9]+' || echo "0")
echo "Detected TensorRT major version: $TRT_VERSION"

if [ "$TRT_VERSION" -ge 10 ]; then
    echo "Using TensorRT 10.x+ syntax (--memPoolSize)"
    WORKSPACE_PARAM="--memPoolSize=workspace:1024"
else
    echo "Using TensorRT 8.x/9.x syntax (--workspace)"
    WORKSPACE_PARAM="--workspace=1024"
fi

# Convert ONNX to TensorRT with FP16 precision
# This is optimized for Jetson Orin NX
$TRTEXEC \
    --onnx="$INPUT_ONNX" \
    --saveEngine="$OUTPUT_ENGINE" \
    --fp16 \
    $WORKSPACE_PARAM \
    --minShapes=images:1x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --maxShapes=images:1x3x640x640 \
    --verbose

echo ""
echo "=========================================="
echo "Conversion completed successfully!"
echo "Engine saved to: $OUTPUT_ENGINE"
echo "=========================================="

# Print usage instructions
echo ""
echo "To use the TensorRT engine in your code:"
echo "  1. Replace 'yolo11n-seg.onnx' with 'yolo11n-seg.engine' in your config"
echo "  2. Or set environment variable: export YOLO_MODEL_PATH=$OUTPUT_ENGINE"

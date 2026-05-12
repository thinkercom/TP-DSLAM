#!/bin/bash
# ======================== experiment ========================
# [EXPERIMENT] Build script for Jetson Orin NX with TensorRT
# Purpose: Compile TP-DSLAM with TensorRT acceleration on Jetson
# Usage: ./scripts/build_jetson.sh
# ======================== experiment end =====================

set -e

echo "=========================================="
echo "TP-DSLAM Jetson Orin NX Build Script"
echo "=========================================="

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "Warning: This script is designed for Jetson devices"
fi

# Create build directory
mkdir -p build
cd build

# Configure with TensorRT enabled
echo "Configuring with TensorRT..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_TENSORRT=ON \
    -DTENSORRT_ROOT=/usr/lib/aarch64-linux-gnu

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "To run KITTI stereo example:"
echo "  ./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTI00-02.yaml /path/to/kitti/sequences/00/"
echo ""
echo "To run TUM RGBD example:"
echo "  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /path/to/TUM/ /path/to/associate.txt"
echo ""
echo "To convert ONNX model to TensorRT:"
echo "  ./scripts/convert_onnx_to_tensorrt.sh yolo11n-seg.onnx yolo11n-seg.engine"

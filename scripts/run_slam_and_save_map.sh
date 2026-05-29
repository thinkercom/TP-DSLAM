#!/bin/bash
# TP-DSLAM SLAM + 3D Map Generation Script
# Usage: ./scripts/run_slam_and_save_map.sh [dataset_type] [dataset_path]

# Default parameters
DATASET_TYPE="${1:-kitti}"
DATASET_PATH="${2:-}"
OUTPUT_DIR="./output_3d_map"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "TP-DSLAM: SLAM + 3D Map Generation"
echo "=========================================="
echo "Dataset Type: $DATASET_TYPE"
echo "Dataset Path: $DATASET_PATH"
echo "Output Dir:   $OUTPUT_DIR"
echo "=========================================="

# Run SLAM based on dataset type
case $DATASET_TYPE in
    kitti)
        if [ -z "$DATASET_PATH" ]; then
            echo "Usage: $0 kitti /path/to/kitti/sequences/XX/"
            exit 1
        fi
        echo "Running Stereo KITTI..."
        ./Examples/Stereo/stereo_kitti \
            ./Vocabulary/ORBvoc.txt \
            ./Examples/Stereo/KITTI00-02.yaml \
            "$DATASET_PATH"
        ;;
    tum)
        if [ -z "$DATASET_PATH" ]; then
            echo "Usage: $0 tum /path/to/TUM_dataset/"
            exit 1
        fi
        echo "Running RGB-D TUM..."
        # Find association file
        ASSOC_FILE=$(find "$DATASET_PATH" -name "*.txt" | head -1)
        ./Examples/RGB-D/rgbd_tum \
            ./Vocabulary/ORBvoc.txt \
            ./Examples/RGB-D/TUM3.yaml \
            "$DATASET_PATH" \
            "$ASSOC_FILE"
        ;;
    euroc)
        if [ -z "$DATASET_PATH" ]; then
            echo "Usage: $0 euroc /path/to/euroc/mav0/"
            exit 1
        fi
        echo "Running Stereo-Inertial EuRoC..."
        ./Examples/Stereo-Inertial/stereo_inertial_euroc \
            ./Vocabulary/ORBvoc.txt \
            ./Examples/Stereo-Inertial/EuRoC.yaml \
            "$DATASET_PATH"
        ;;
    *)
        echo "Unknown dataset type: $DATASET_TYPE"
        echo "Supported types: kitti, tum, euroc"
        exit 1
        ;;
esac

# Check if point cloud was generated
if [ -f "MapPoints.pcd" ]; then
    echo ""
    echo "=========================================="
    echo "3D Map Generated Successfully!"
    echo "=========================================="
    
    # Move to output directory
    mv MapPoints.pcd "$OUTPUT_DIR/map_${TIMESTAMP}.pcd"
    mv CameraTrajectory.txt "$OUTPUT_DIR/trajectory_${TIMESTAMP}.txt" 2>/dev/null
    
    echo "Files saved to:"
    echo "  Point Cloud: $OUTPUT_DIR/map_${TIMESTAMP}.pcd"
    echo "  Trajectory:  $OUTPUT_DIR/trajectory_${TIMESTAMP}.txt"
    echo ""
    echo "To visualize the 3D map:"
    echo "  python3 scripts/visualize_map.py $OUTPUT_DIR/map_${TIMESTAMP}.pcd"
    echo ""
    echo "Or convert to PLY for other viewers:"
    echo "  python3 -c \"import open3d; pcd=open3d.io.read_point_cloud('$OUTPUT_DIR/map_${TIMESTAMP}.pcd'); open3d.io.write_point_cloud('map.ply', pcd)\""
    echo "=========================================="
else
    echo ""
    echo "Warning: MapPoints.pcd not found. Check SLAM output for errors."
fi

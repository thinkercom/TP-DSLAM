# TP-DSLAM 3D Reconstruction Guide

This guide explains how to create high-quality 3D reconstructions from TP-DSLAM output.

## Overview of Methods

| Method | Quality | Speed | Use Case |
|--------|---------|-------|----------|
| Sparse Point Cloud | Low | Fast | Quick preview |
| Dense Point Cloud | Medium | Medium | General purpose |
| TSDF Mesh | High | Slow | High-quality reconstruction |

---

## Method 1: Sparse Point Cloud (Built-in)

This is the default method, automatically saves `MapPoints.pcd` after SLAM.

```bash
# Run SLAM (auto-saves sparse point cloud)
./Examples/Stereo/stereo_kitti ./Vocabulary/ORBvoc.txt ./Examples/Stereo/KITTI00-02.yaml /path/to/kitti/sequences/00/

# Visualize
python3 scripts/visualize_map.py MapPoints.pcd
```

**Pros:** Fast, automatic
**Cons:** Very sparse, low quality

---

## Method 2: Dense Point Cloud Reconstruction

Creates dense point cloud by fusing depth maps from all frames.

### For RGB-D Dataset (TUM)
```bash
python3 scripts/dense_reconstruction.py \
    --type rgbd \
    --path /path/to/TUM/rgbd_dataset_freiburg3_walking_xyz \
    --trajectory CameraTrajectory.txt \
    --output DenseMap.ply \
    --max_frames 200
```

### For Stereo Dataset (KITTI)
```bash
python3 scripts/dense_reconstruction.py \
    --type kitti \
    --path /path/to/kitti/sequences/00 \
    --trajectory CameraTrajectory.txt \
    --output DenseMap.ply \
    --max_frames 100
```

### Visualize
```bash
python3 scripts/visualize_map.py DenseMap.ply
```

---

## Method 3: TSDF Volume Fusion (Recommended)

Uses Truncated Signed Distance Function for high-quality mesh reconstruction.

### For RGB-D Dataset (TUM)
```bash
python3 scripts/tsdf_fusion.py \
    --type rgbd \
    --path /path/to/TUM/rgbd_dataset_freiburg3_walking_xyz \
    --trajectory CameraTrajectory.txt \
    --output DenseMesh.ply \
    --max_frames 200 \
    --visualize
```

### For Stereo Dataset (KITTI)
```bash
python3 scripts/tsdf_fusion.py \
    --type kitti \
    --path /path/to/kitti/sequences/00 \
    --trajectory CameraTrajectory.txt \
    --output DenseMesh.ply \
    --max_frames 100 \
    --visualize
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--voxel_size` | Voxel size in meters (smaller = finer detail) | 0.01 (indoor) / 0.05 (outdoor) |
| `--max_frames` | Maximum frames to process | 100 |
| `--visualize` | Show result after reconstruction | False |

---

## Post-Processing

### Using Open3D
```python
import open3d as o3d

# Load mesh
mesh = o3d.io.read_triangle_mesh("DenseMesh.ply")
mesh.compute_vertex_normals()

# Simplify mesh (reduce triangles)
mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=100000)

# Save simplified mesh
o3d.io.write_triangle_mesh("DenseMesh_simplified.ply", mesh_simplified)

# Visualize
o3d.visualization.draw_geometries([mesh_simplified])
```

### Using MeshLab
1. Open MeshLab
2. File → Import Mesh → Select `DenseMesh.ply`
3. Filters → Cleaning → Remove duplicate faces/vertices
4. Filters → Remeshing → Quadric Edge Collapse Decimation

### Using CloudCompare
1. Open CloudCompare
2. File → Open → Select PLY file
3. Tools → Registration → Align with ground truth

---

## Advanced: Custom Reconstruction Pipeline

### Step 1: Extract keyframe poses
```bash
# After SLAM, save keyframe trajectory
cp KeyFrameTrajectory.txt keyframes.txt
```

### Step 2: Run TSDF fusion with keyframes only
```bash
python3 scripts/tsdf_fusion.py \
    --type rgbd \
    --path /path/to/dataset \
    --trajectory KeyFrameTrajectory.txt \
    --output KeyframeMesh.ply \
    --max_frames 50
```

### Step 3: Texture mapping (optional)
```bash
# Using MVS (Multi-View Stereo) for texture
# Install COLMAP or OpenMVS for advanced texturing
```

---

## Troubleshooting

### Problem: Reconstruction is too sparse
**Solution:** Increase `--max_frames` or decrease `--voxel_size`

### Problem: Reconstruction has holes
**Solution:** 
- Ensure good camera coverage
- Increase `sdf_trunc` parameter
- Use more frames

### Problem: Colors are wrong
**Solution:** Check RGB/depth alignment, ensure correct intrinsics

### Problem: Scale is wrong
**Solution:** Verify depth scale factor (1000 for TUM, varies for other datasets)

---

## Example Results

### Before (Sparse)
- Only ORB feature points
- ~1000-5000 points
- No surface information

### After TSDF Fusion (Dense)
- Complete surface reconstruction
- ~1M+ points or mesh vertices
- Textured surfaces
- Smooth, continuous geometry

---

## Dependencies

```bash
# Required
pip install open3d numpy opencv-python

# Optional (for advanced processing)
pip install trimesh pyvista
```

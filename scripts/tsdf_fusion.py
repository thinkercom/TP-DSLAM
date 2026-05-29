#!/usr/bin/env python3
"""
TP-DSLAM TSDF Volume Fusion for High-Quality 3D Reconstruction
Uses Open3D's TSDF integration for dense, smooth reconstruction

Usage:
    python3 scripts/tsdf_fusion.py --type rgbd --path /path/to/TUM_dataset --trajectory CameraTrajectory.txt
"""

import argparse
import numpy as np
import cv2
import os
import glob
import open3d as o3d
from pathlib import Path

def load_poses_tum(trajectory_file):
    """Load camera poses from TUM format trajectory file"""
    poses = []
    timestamps = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                values = line.strip().split()
                if len(values) == 8:
                    t = float(values[0])
                    tx, ty, tz = float(values[1]), float(values[2]), float(values[3])
                    qw, qx, qy, qz = float(values[4]), float(values[5]), float(values[6]), float(values[7])
                    
                    # Convert quaternion to rotation matrix
                    R = o3d.geometry.get_rotation_matrix_from_quaternion([qw, qx, qy, qz])
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = [tx, ty, tz]
                    
                    poses.append(pose)
                    timestamps.append(t)
    return poses, timestamps

def load_poses_kitti(trajectory_file):
    """Load camera poses from KITTI format trajectory file"""
    poses = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                values = line.strip().split()
                if len(values) == 12:
                    pose = np.eye(4)
                    pose[:3, :] = np.array(values, dtype=float).reshape(3, 4)
                    poses.append(pose)
    return poses

def rgbd_tsdf_fusion(dataset_path, trajectory_file, output_file, max_frames=200):
    """Perform TSDF fusion for RGB-D dataset"""
    print("Starting RGB-D TSDF Fusion...")
    
    # Find RGB and depth images
    rgb_files = sorted(glob.glob(os.path.join(dataset_path, "rgb/*.png")))
    depth_files = sorted(glob.glob(os.path.join(dataset_path, "depth/*.png")))
    
    if not rgb_files:
        rgb_files = sorted(glob.glob(os.path.join(dataset_path, "color/*.png")))
    
    # Load poses
    poses, timestamps = load_poses_tum(trajectory_file)
    
    # Camera intrinsics (TUM RGB-D)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640, height=480,
        fx=525.0, fy=525.0, cx=319.5, cy=239.5
    )
    
    # Initialize TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,  # 1cm voxels
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    num_frames = min(len(rgb_files), len(depth_files), len(poses), max_frames)
    step = max(1, len(rgb_files) // num_frames)
    
    print(f"Processing {num_frames} frames...")
    
    for i in range(0, len(rgb_files), step):
        frame_idx = min(i, len(poses) - 1)
        print(f"Integrating frame {i}...")
        
        # Load images
        color = o3d.io.read_image(rgb_files[i])
        depth = o3d.io.read_image(depth_files[i])
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth,
            depth_scale=1000.0,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False
        )
        
        # Get camera pose (world to camera)
        pose = poses[frame_idx]
        extrinsic = np.linalg.inv(pose)
        
        # Integrate into TSDF volume
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    # Extract mesh
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # Save mesh
    o3d.io.write_triangle_mesh(output_file, mesh)
    print(f"Saved mesh to {output_file}")
    
    # Also extract point cloud
    pcd_file = output_file.replace('.ply', '_pcd.ply').replace('.obj', '_pcd.ply')
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(pcd_file, pcd)
    print(f"Saved point cloud to {pcd_file}")
    
    return mesh, pcd

def stereo_tsdf_fusion(dataset_path, trajectory_file, output_file, max_frames=100):
    """Perform TSDF fusion for stereo dataset (KITTI)"""
    print("Starting Stereo TSDF Fusion...")
    
    # Find stereo images
    left_files = sorted(glob.glob(os.path.join(dataset_path, "image_0/*.png")))
    right_files = sorted(glob.glob(os.path.join(dataset_path, "image_1/*.png")))
    
    # Load poses
    poses = load_poses_kitti(trajectory_file)
    
    # Camera intrinsics (KITTI)
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=1241, height=376,
        fx=718.856, fy=718.856, cx=607.1928, cy=185.2157
    )
    
    # Initialize TSDF volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.05,  # 5cm voxels for outdoor
        sdf_trunc=0.15,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # Stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    baseline = 0.53716
    fx = 718.856
    
    num_frames = min(len(left_files), len(right_files), len(poses), max_frames)
    step = max(1, len(left_files) // num_frames)
    
    print(f"Processing {num_frames} frames...")
    
    for i in range(0, len(left_files), step):
        frame_idx = min(i, len(poses) - 1)
        print(f"Integrating frame {i}...")
        
        # Load stereo images
        img_left = cv2.imread(left_files[i])
        img_right = cv2.imread(right_files[i])
        
        if img_left is None or img_right is None:
            continue
        
        # Compute disparity
        disparity = stereo.compute(
            cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        ).astype(np.float32) / 16.0
        
        # Convert to depth
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = (fx * baseline) / disparity[valid]
        depth = (depth * 1000).astype(np.uint16)  # Convert to mm
        
        # Convert to Open3D images
        color_o3d = o3d.geometry.Image(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth)
        
        # Create RGBD image
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d,
            depth_scale=1000.0,
            depth_trunc=50.0,
            convert_rgb_to_intensity=False
        )
        
        # Get camera pose
        pose = poses[frame_idx]
        extrinsic = np.linalg.inv(pose)
        
        # Integrate into TSDF volume
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    # Extract mesh
    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # Save mesh
    o3d.io.write_triangle_mesh(output_file, mesh)
    print(f"Saved mesh to {output_file}")
    
    # Also extract point cloud
    pcd_file = output_file.replace('.ply', '_pcd.ply').replace('.obj', '_pcd.ply')
    pcd = volume.extract_point_cloud()
    o3d.io.write_point_cloud(pcd_file, pcd)
    print(f"Saved point cloud to {pcd_file}")
    
    return mesh, pcd

def visualize_reconstruction(mesh_file):
    """Visualize the reconstructed mesh"""
    print(f"Loading mesh: {mesh_file}")
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    
    print("Visualizing... (press 'q' to quit)")
    o3d.visualization.draw_geometries([mesh],
                                       window_name="TP-DSLAM 3D Reconstruction",
                                       width=1920, height=1080,
                                       mesh_show_back_face=True)

def main():
    parser = argparse.ArgumentParser(description='TP-DSLAM TSDF Fusion')
    parser.add_argument('--type', type=str, required=True, choices=['rgbd', 'stereo', 'kitti'],
                        help='Dataset type')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--trajectory', type=str, default='CameraTrajectory.txt',
                        help='Path to trajectory file')
    parser.add_argument('--output', type=str, default='DenseMesh.ply',
                        help='Output mesh file (PLY or OBJ)')
    parser.add_argument('--max_frames', type=int, default=100,
                        help='Maximum number of frames to process')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize result after reconstruction')
    parser.add_argument('--voxel_size', type=float, default=None,
                        help='Voxel size for TSDF volume (meters)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TP-DSLAM TSDF Volume Fusion")
    print("=" * 60)
    print(f"Dataset: {args.path}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Output: {args.output}")
    print(f"Max Frames: {args.max_frames}")
    print("=" * 60)
    
    if args.type == 'rgbd':
        mesh, pcd = rgbd_tsdf_fusion(args.path, args.trajectory, args.output, args.max_frames)
    elif args.type in ['stereo', 'kitti']:
        mesh, pcd = stereo_tsdf_fusion(args.path, args.trajectory, args.output, args.max_frames)
    
    print("\n" + "=" * 60)
    print("Reconstruction Complete!")
    print("=" * 60)
    print(f"Mesh: {args.output}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    print(f"Point Cloud: {args.output.replace('.ply', '_pcd.ply')}")
    print(f"Points: {len(pcd.points)}")
    
    if args.visualize:
        visualize_reconstruction(args.output)
    else:
        print("\nTo visualize:")
        print(f"  python3 -c \"import open3d as o3d; mesh=o3d.io.read_triangle_mesh('{args.output}'); mesh.compute_vertex_normals(); o3d.visualization.draw_geometries([mesh])\"")

if __name__ == "__main__":
    main()

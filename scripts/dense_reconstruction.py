#!/usr/bin/env python3
"""
TP-DSLAM Dense 3D Reconstruction
Uses depth maps to create dense point cloud or mesh

Usage:
    python3 scripts/dense_reconstruction.py --type rgbd --path /path/to/TUM_dataset
    python3 scripts/dense_reconstruction.py --type kitti --path /path/to/kitti/sequences/00
"""

import argparse
import numpy as np
import cv2
import os
import glob
from pathlib import Path

def load_poses(trajectory_file):
    """Load camera poses from trajectory file"""
    poses = []
    with open(trajectory_file, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                values = line.strip().split()
                if len(values) == 12:  # KITTI format
                    pose = np.eye(4)
                    pose[:3, :] = np.array(values, dtype=float).reshape(3, 4)
                    poses.append(pose)
                elif len(values) == 8:  # TUM format
                    tx, ty, tz = float(values[1]), float(values[2]), float(values[3])
                    qw, qx, qy, qz = float(values[4]), float(values[5]), float(values[6]), float(values[7])
                    # Convert quaternion to rotation matrix
                    R = quat_to_rotmat(qw, qx, qy, qz)
                    pose = np.eye(4)
                    pose[:3, :3] = R
                    pose[:3, 3] = [tx, ty, tz]
                    poses.append(pose)
    return poses

def quat_to_rotmat(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix"""
    R = np.array([
        [1 - 2*qy*qy - 2*qz*qz, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy]
    ])
    return R

def load_stereo_pair(left_path, right_path):
    """Load stereo image pair"""
    img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    return img_left, img_right

def compute_depth_stereo(img_left, img_right, baseline=0.53716, fx=718.856):
    """Compute depth map using stereo matching"""
    # StereoSGBM matcher
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
    
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    
    # Convert disparity to depth
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = (fx * baseline) / disparity[valid]
    
    return depth

def create_point_cloud_from_rgbd(color, depth, intrinsics, pose, depth_scale=1000.0, depth_trunc=3.0):
    """Create point cloud from RGB-D image"""
    h, w = depth.shape
    
    # Create mesh grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Depth values
    z = depth.astype(np.float32) / depth_scale
    
    # Filter valid depths
    valid = (z > 0) & (z < depth_trunc)
    
    # Back-project to 3D
    x = (u - intrinsics['cx']) * z / intrinsics['fx']
    y = (v - intrinsics['cy']) * z / intrinsics['fy']
    
    # Stack points
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    
    # Transform to world coordinates
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = (R @ points.T).T + t
    
    # Get colors
    if color is not None:
        colors = color[valid] / 255.0
    else:
        colors = np.ones_like(points_world) * 0.5
    
    return points_world, colors

def create_point_cloud_from_stereo(img_left, depth, intrinsics, pose):
    """Create point cloud from stereo depth"""
    h, w = depth.shape
    
    # Create mesh grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Filter valid depths
    valid = (depth > 0) & (depth < 50)  # Max 50m for KITTI
    
    # Back-project to 3D
    z = depth[valid]
    x = (u[valid] - intrinsics['cx']) * z / intrinsics['fx']
    y = (v[valid] - intrinsics['cy']) * z / intrinsics['fy']
    
    points = np.stack([x, y, z], axis=-1)
    
    # Transform to world coordinates
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = (R @ points.T).T + t
    
    # Get colors from left image
    if img_left is not None:
        colors = cv2.cvtColor(img_left, cv2.COLOR_GRAY2RGB)[valid] / 255.0
    else:
        colors = np.ones_like(points_world) * 0.5
    
    return points_world, colors

def process_rgbd_dataset(dataset_path, trajectory_file, output_file, max_frames=100):
    """Process RGB-D dataset (TUM format)"""
    print(f"Processing RGB-D dataset: {dataset_path}")
    
    # Load association file or find images
    rgb_files = sorted(glob.glob(os.path.join(dataset_path, "rgb/*.png")))
    depth_files = sorted(glob.glob(os.path.join(dataset_path, "depth/*.png")))
    
    if not rgb_files:
        rgb_files = sorted(glob.glob(os.path.join(dataset_path, "*.png")))[:len(depth_files)]
    
    # Load poses
    poses = load_poses(trajectory_file) if os.path.exists(trajectory_file) else [np.eye(4)]
    
    # Camera intrinsics (TUM dataset)
    intrinsics = {
        'fx': 525.0,
        'fy': 525.0,
        'cx': 319.5,
        'cy': 239.5
    }
    
    all_points = []
    all_colors = []
    
    # Process frames
    num_frames = min(len(rgb_files), len(depth_files), len(poses), max_frames)
    step = max(1, len(rgb_files) // num_frames)
    
    for i in range(0, len(rgb_files), step):
        if len(all_points) > 0 and len(all_points[-1]) > 1000000:  # 1M points limit
            break
            
        print(f"Processing frame {i}/{len(rgb_files)}...")
        
        # Load images
        color = cv2.imread(rgb_files[i])
        depth = cv2.imread(depth_files[i], cv2.IMREAD_UNCHANGED)
        
        if color is None or depth is None:
            continue
        
        # Get pose
        pose_idx = min(i, len(poses) - 1)
        pose = poses[pose_idx]
        
        # Create point cloud
        points, colors = create_point_cloud_from_rgbd(color, depth, intrinsics, pose)
        
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
    
    # Merge all points
    if all_points:
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        # Save point cloud
        save_point_cloud_ply(output_file, all_points, all_colors)
        print(f"Saved {len(all_points)} points to {output_file}")
    else:
        print("No points generated!")

def process_kitti_dataset(dataset_path, trajectory_file, output_file, max_frames=100):
    """Process KITTI stereo dataset"""
    print(f"Processing KITTI dataset: {dataset_path}")
    
    # Find stereo images
    left_files = sorted(glob.glob(os.path.join(dataset_path, "image_0/*.png")))
    right_files = sorted(glob.glob(os.path.join(dataset_path, "image_1/*.png")))
    
    # Load poses
    poses = load_poses(trajectory_file) if os.path.exists(trajectory_file) else [np.eye(4)]
    
    # Camera intrinsics (KITTI)
    intrinsics = {
        'fx': 718.856,
        'fy': 718.856,
        'cx': 607.1928,
        'cy': 185.2157
    }
    baseline = 0.53716
    
    all_points = []
    all_colors = []
    
    # Process frames
    num_frames = min(len(left_files), len(right_files), len(poses), max_frames)
    step = max(1, len(left_files) // num_frames)
    
    for i in range(0, len(left_files), step):
        if len(all_points) > 0 and len(all_points[-1]) > 1000000:  # 1M points limit
            break
            
        print(f"Processing frame {i}/{len(left_files)}...")
        
        # Load stereo pair
        img_left, img_right = load_stereo_pair(left_files[i], right_files[i])
        
        if img_left is None or img_right is None:
            continue
        
        # Compute depth
        depth = compute_depth_stereo(img_left, img_right, baseline, intrinsics['fx'])
        
        # Get pose
        pose_idx = min(i, len(poses) - 1)
        pose = poses[pose_idx]
        
        # Create point cloud
        points, colors = create_point_cloud_from_stereo(img_left, depth, intrinsics, pose)
        
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
    
    # Merge all points
    if all_points:
        all_points = np.vstack(all_points)
        all_colors = np.vstack(all_colors)
        
        # Save point cloud
        save_point_cloud_ply(output_file, all_points, all_colors)
        print(f"Saved {len(all_points)} points to {output_file}")
    else:
        print("No points generated!")

def save_point_cloud_ply(filename, points, colors):
    """Save point cloud to PLY format"""
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(np.uint8)
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")

def main():
    parser = argparse.ArgumentParser(description='TP-DSLAM Dense 3D Reconstruction')
    parser.add_argument('--type', type=str, required=True, choices=['rgbd', 'kitti', 'stereo'],
                        help='Dataset type')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to dataset')
    parser.add_argument('--trajectory', type=str, default='CameraTrajectory.txt',
                        help='Path to trajectory file')
    parser.add_argument('--output', type=str, default='DenseMap.ply',
                        help='Output PLY file')
    parser.add_argument('--max_frames', type=int, default=100,
                        help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TP-DSLAM Dense 3D Reconstruction")
    print("=" * 60)
    
    if args.type == 'rgbd':
        process_rgbd_dataset(args.path, args.trajectory, args.output, args.max_frames)
    elif args.type in ['kitti', 'stereo']:
        process_kitti_dataset(args.path, args.trajectory, args.output, args.max_frames)
    
    print("\nTo visualize the dense map:")
    print(f"  python3 scripts/visualize_map.py {args.output}")
    print("  or open with MeshLab/CloudCompare")

if __name__ == "__main__":
    main()

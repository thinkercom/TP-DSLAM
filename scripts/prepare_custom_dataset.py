#!/usr/bin/env python3
"""
TP-DSLAM Custom Dataset Preparation Tool
Helps prepare custom datasets for 3D reconstruction

Usage:
    python3 scripts/prepare_custom_dataset.py --input /path/to/images --output /path/to/output
"""

import argparse
import cv2
import numpy as np
import os
import json
from pathlib import Path
import glob

def create_dataset_structure(output_path, dataset_name="custom"):
    """Create standard dataset directory structure"""
    dirs = [
        os.path.join(output_path, dataset_name, "rgb"),
        os.path.join(output_path, dataset_name, "depth"),
        os.path.join(output_path, dataset_name, "masks"),
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"Created: {d}")
    
    return os.path.join(output_path, dataset_name)

def create_camera_config(output_path, camera_type="pinhole", width=640, height=480, 
                         fx=500, fy=500, cx=320, cy=240, depth_scale=1000):
    """Create camera configuration file"""
    config = {
        "camera": {
            "type": camera_type,
            "width": width,
            "height": height,
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "depth_scale": depth_scale
        },
        "dataset": {
            "name": "custom",
            "rgb_dir": "rgb",
            "depth_dir": "depth",
            "mask_dir": "masks"
        }
    }
    
    config_path = os.path.join(output_path, "camera_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Created camera config: {config_path}")
    return config_path

def extract_frames_from_video(video_path, output_path, fps=10, prefix="frame"):
    """Extract frames from video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps / fps)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            filename = f"{prefix}_{saved_count:06d}.png"
            filepath = os.path.join(output_path, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from video")

def create_synthetic_depth_from_stereo(left_path, right_path, output_path, 
                                        fx=500, baseline=0.1):
    """Create depth map from stereo pair"""
    img_left = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    
    if img_left is None or img_right is None:
        return
    
    # StereoSGBM
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )
    
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    
    # Convert to depth
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = (fx * baseline) / disparity[valid]
    
    # Scale to uint16
    depth_scaled = (depth * 1000).astype(np.uint16)
    
    cv2.imwrite(output_path, depth_scaled)

def create_association_file(rgb_dir, depth_dir, output_path):
    """Create TUM-format association file"""
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    
    with open(output_path, 'w') as f:
        for i, (rgb, depth) in enumerate(zip(rgb_files, depth_files)):
            timestamp = i * 0.033  # Assume 30fps
            rgb_name = os.path.basename(rgb)
            depth_name = os.path.basename(depth)
            f.write(f"{timestamp:.6f} rgb/{rgb_name} {timestamp:.6f} depth/{depth_name}\n")
    
    print(f"Created association file: {output_path}")

def process_single_camera_images(input_dir, output_path, target_size=(640, 480)):
    """Process images from single camera (no depth)"""
    image_files = sorted(glob.glob(os.path.join(input_dir, "*.jpg")) + 
                         glob.glob(os.path.join(input_dir, "*.png")) +
                         glob.glob(os.path.join(input_dir, "*.jpeg")))
    
    rgb_dir = os.path.join(output_path, "rgb")
    
    print(f"Processing {len(image_files)} images...")
    
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Resize
        img_resized = cv2.resize(img, target_size)
        
        # Save
        output_file = os.path.join(rgb_dir, f"frame_{i:06d}.png")
        cv2.imwrite(output_file, img_resized)
    
    print(f"Processed {len(image_files)} images")

def process_rgbd_pairs(rgb_dir, depth_dir, output_path, target_size=(640, 480)):
    """Process RGB-D image pairs"""
    rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "*.png")) + 
                       glob.glob(os.path.join(rgb_dir, "*.jpg")))
    depth_files = sorted(glob.glob(os.path.join(depth_dir, "*.png")))
    
    out_rgb_dir = os.path.join(output_path, "rgb")
    out_depth_dir = os.path.join(output_path, "depth")
    
    print(f"Processing {len(rgb_files)} RGB-D pairs...")
    
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
        # Process RGB
        rgb = cv2.imread(rgb_path)
        if rgb is not None:
            rgb_resized = cv2.resize(rgb, target_size)
            cv2.imwrite(os.path.join(out_rgb_dir, f"frame_{i:06d}.png"), rgb_resized)
        
        # Process Depth
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is not None:
            depth_resized = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(out_depth_dir, f"frame_{i:06d}.png"), depth_resized)

def main():
    parser = argparse.ArgumentParser(description='TP-DSLAM Custom Dataset Preparation')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with images/videos')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for prepared dataset')
    parser.add_argument('--type', type=str, default='rgbd',
                        choices=['rgbd', 'stereo', 'video', 'single'],
                        help='Input data type')
    parser.add_argument('--name', type=str, default='custom_dataset',
                        help='Dataset name')
    parser.add_argument('--width', type=int, default=640,
                        help='Target image width')
    parser.add_argument('--height', type=int, default=480,
                        help='Target image height')
    parser.add_argument('--fps', type=int, default=10,
                        help='Frame extraction rate for video')
    parser.add_argument('--fx', type=float, default=500,
                        help='Camera focal length x')
    parser.add_argument('--fy', type=float, default=500,
                        help='Camera focal length y')
    parser.add_argument('--cx', type=float, default=320,
                        help='Camera principal point x')
    parser.add_argument('--cy', type=float, default=240,
                        help='Camera principal point y')
    parser.add_argument('--depth_scale', type=float, default=1000,
                        help='Depth scale factor')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TP-DSLAM Custom Dataset Preparation")
    print("=" * 60)
    
    # Create output directory structure
    dataset_path = create_dataset_structure(args.output, args.name)
    
    # Create camera config
    create_camera_config(dataset_path, 
                        width=args.width, height=args.height,
                        fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
                        depth_scale=args.depth_scale)
    
    # Process based on input type
    if args.type == 'video':
        extract_frames_from_video(args.input, os.path.join(dataset_path, "rgb"), fps=args.fps)
    
    elif args.type == 'single':
        process_single_camera_images(args.input, dataset_path, (args.width, args.height))
    
    elif args.type == 'rgbd':
        # Assume input has rgb/ and depth/ subdirectories
        process_rgbd_pairs(args.input, args.input, dataset_path, (args.width, args.height))
    
    elif args.type == 'stereo':
        # For stereo, we'll create depth from stereo pairs later
        print("Stereo processing: Use stereo reconstruction script")
    
    # Create association file
    create_association_file(
        os.path.join(dataset_path, "rgb"),
        os.path.join(dataset_path, "depth"),
        os.path.join(dataset_path, "association.txt")
    )
    
    print("\n" + "=" * 60)
    print("Dataset Preparation Complete!")
    print("=" * 60)
    print(f"Dataset location: {dataset_path}")
    print(f"\nNext steps:")
    print(f"1. Run SLAM:")
    print(f"   ./Examples/RGB-D/rgbd_tum ./Vocabulary/ORBvoc.txt \\")
    print(f"       ./Examples/RGB-D/TUM3.yaml {dataset_path} {dataset_path}/association.txt")
    print(f"\n2. Run 3D reconstruction:")
    print(f"   python3 scripts/tsdf_fusion.py --type rgbd --path {dataset_path}")

if __name__ == "__main__":
    main()

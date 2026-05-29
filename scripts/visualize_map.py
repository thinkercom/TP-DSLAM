#!/usr/bin/env python3
"""
TP-DSLAM 3D Map Visualization Script
Usage: python3 scripts/visualize_map.py [path_to_pcd_file]
"""

import sys
import numpy as np

def visualize_with_open3d(pcd_file):
    """Visualize PCD file using Open3D"""
    try:
        import open3d as o3d
        print(f"Loading point cloud: {pcd_file}")
        pcd = o3d.io.read_point_cloud(pcd_file)
        print(f"Loaded {len(pcd.points)} points")
        
        # Estimate normals for better visualization
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
        
        # Visualize
        print("Visualizing... (press 'q' to quit)")
        o3d.visualization.draw_geometries([pcd], 
                                           window_name="TP-DSLAM 3D Map",
                                           width=1280, height=720,
                                           point_show_normal=False)
    except ImportError:
        print("Open3D not installed. Install with: pip install open3d")
        return False
    return True

def visualize_with_mayavi(pcd_file):
    """Visualize PCD file using Mayavi"""
    try:
        from mayavi import mlab
        import struct
        
        # Read PCD file manually (simple version)
        points = []
        with open(pcd_file, 'rb') as f:
            # Skip header
            line = f.readline()
            while not line.startswith(b'DATA'):
                line = f.readline()
            
            # Read binary data
            while True:
                data = f.read(12)  # 3 floats * 4 bytes
                if len(data) < 12:
                    break
                x, y, z = struct.unpack('fff', data)
                points.append([x, y, z])
        
        points = np.array(points)
        print(f"Loaded {len(points)} points")
        
        # Visualize
        mlab.points3d(points[:, 0], points[:, 1], points[:, 2], 
                      mode='point', colormap='viridis')
        mlab.title("TP-DSLAM 3D Map")
        mlab.show()
    except ImportError:
        print("Mayavi not installed. Install with: pip install mayavi")
        return False
    return True

def visualize_with_pcl_viewer(pcd_file):
    """Use PCL viewer (command line)"""
    import subprocess
    try:
        subprocess.run(['pcl_viewer', pcd_file], check=True)
    except FileNotFoundError:
        print("PCL viewer not found. Install PCL or use alternative visualization.")
        return False
    return True

def convert_to_ply(pcd_file):
    """Convert PCD to PLY format for broader compatibility"""
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(pcd_file)
        ply_file = pcd_file.replace('.pcd', '.ply')
        o3d.io.write_point_cloud(ply_file, pcd)
        print(f"Converted to PLY: {ply_file}")
        return ply_file
    except ImportError:
        print("Open3D not installed for conversion.")
        return None

def main():
    if len(sys.argv) < 2:
        pcd_file = "MapPoints.pcd"
        print(f"No file specified, using default: {pcd_file}")
    else:
        pcd_file = sys.argv[1]
    
    print("=" * 50)
    print("TP-DSLAM 3D Map Visualization")
    print("=" * 50)
    print(f"Point cloud file: {pcd_file}")
    print()
    
    # Try different visualization methods
    if not visualize_with_open3d(pcd_file):
        if not visualize_with_mayavi(pcd_file):
            if not visualize_with_pcl_viewer(pcd_file):
                print("\nFallback: Converting to PLY format...")
                ply_file = convert_to_ply(pcd_file)
                if ply_file:
                    print(f"Open {ply_file} with any 3D viewer (MeshLab, CloudCompare, etc.)")

if __name__ == "__main__":
    main()

# TP-DSLAM: Robust Dynamic SLAM with Temporal-Semantic Fusion and Geometric Gating

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Ubuntu_20.04%20%7C%2022.04-orange.svg)](https://ubuntu.com/)
[![Language](https://img.shields.io/badge/language-C++-blue.svg)](https://isocpp.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.x%20%7C%2012.x-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**TP-DSLAM** is a robust visual SLAM system specifically designed for **dynamic environments**. By integrating **Temporal-Semantic Fusion** and a **Geometric Disparity Gating** mechanism, our system effectively mitigates the interference of dynamic objects (e.g., walking persons, moving vehicles) and maintains high-precision localization even under challenging conditions like fast motion and lighting variations.

> **Note:** This repository contains the source code for the paper *[TP-DSLAM: Robust Dynamic SLAM with Temporal-Semantic Fusion and Geometric Gating]*.

<p align="center">
  <img src="https://img.shields.io/badge/SLAM-Dynamic%20Robustness-brightgreen" alt="Dynamic SLAM">
  <img src="https://img.shields.io/badge/Sensor-Multi--Modal-blue" alt="Multi-Modal">
  <img src="https://img.shields.io/badge/Inference-ONNX%20%7C%20TensorRT-orange" alt="Inference">
</p>

---

## Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Performance Evaluation](#-performance-evaluation)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Running the System](#-running-the-system)
- [Configuration](#-configuration)
- [Project Structure](#-project-structure)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## Key Features

### Dynamic Object Robustness

- **Deep Learning-based Semantic Segmentation**: Integrates YOLO-based instance segmentation via ONNX Runtime for real-time dynamic object detection
- **Time-Semantic Cache**: Maintains temporal consistency across frames to handle occlusions and temporary dynamic noise
- **Dynamic Prior Maps**: Generates pixel-level dynamic probability maps for precise feature filtering

### Geometric Consistency

- **Disparity Gating Mechanism**: Filters outliers based on geometric constraints, significantly improving tracking stability
- **Dynamic Confidence Weighting**: Employs a soft-weighting strategy to preserve useful features while suppressing dynamic noise
- **Pose Smoothing**: Applies motion prediction and smoothing for robust pose estimation

### System Capabilities

- **Multi-Modal Support**: Monocular, Stereo, RGB-D, and their Inertial variants
- **Real-time Performance**: Optimized to run efficiently on standard laptop hardware (tested on 13th Gen Intel Core i5)
- **Edge Deployment**: Supports TensorRT acceleration for NVIDIA Jetson devices
- **Comprehensive Dataset Support**: Pre-configured for TUM RGB-D, KITTI, EuRoC, VIODE, ETH3D, and TUM-VI datasets

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          TP-DSLAM System                           │
├─────────────────────────────────────────────────────────────────────┤
│  Input Stream (Mono/Stereo/RGB-D + IMU)                           │
│         ↓                                                           │
│  ┌─────────────────┐    ┌─────────────────┐                       │
│  │  ORB Extractor   │    │  YOLO Segmentation │                       │
│  │  (Feature Points) │    │  (Dynamic Detection)│                       │
│  └────────┬────────┘    └────────┬────────┘                       │
│           ↓                      ↓                                 │
│  ┌─────────────────────────────────────────┐                       │
│  │     Temporal-Semantic Fusion Module     │                       │
│  │  (Dynamic Prior Map Generation)         │                       │
│  └────────────────────┬────────────────────┘                       │
│                       ↓                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │      Geometric Disparity Gating         │                       │
│  │  (Dynamic Keypoint Filtering)           │                       │
│  └────────────────────┬────────────────────┘                       │
│                       ↓                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │      Tracking & Local Mapping           │                       │
│  │  (Pose Estimation & Map Update)         │                       │
│  └────────────────────┬────────────────────┘                       │
│                       ↓                                             │
│  ┌─────────────────────────────────────────┐                       │
│  │      Loop Closing & Map Optimization    │                       │
│  └─────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Performance Evaluation

We evaluated TP-DSLAM on standard benchmarks including **TUM RGB-D**, **KITTI**, **EuRoC**, and **VIODE**. Our system demonstrates state-of-the-art performance, particularly in highly dynamic scenarios.

### Absolute Trajectory Error (RMSE) on TUM RGB-D (fr3/walking sequences)

| Sequence | ORB-SLAM3 | DynaSLAM | YDD-SLAM | **TP-DSLAM (Ours)** |
|:---|:---:|:---:|:---:|:---:|
| **w_half** | 0.312 | 0.025 | 0.028 | **0.026** |
| **w_xyz** | 0.510 | **0.015** | **0.015** | **0.015** |
| **w_rpy** | 0.536 | **0.035** | 0.036 | **0.035** |
| **Avg. (Dyn)** | 0.381 | 0.020 | 0.022 | **0.021** |

### Robustness in Challenging Conditions

| Dataset | Sequence | ORB-SLAM3 | TP-DSLAM |
|:---|:---|:---:|:---:|
| **EuRoC** | V203 (Fast Motion) | 0.233m | **0.077m** |
| **VIODE** | City_night_High | >2.7m (Failed) | **0.477m** |

---

## Prerequisites

### System Requirements

- **Operating System**: Ubuntu 20.04 / 22.04
- **C++ Compiler**: GCC >= 7.5 (Support for C++14)
- **CMake**: >= 3.10
- **CUDA** (Optional): >= 11.0 for TensorRT support

### Dependencies

| Library | Version | Description |
|:---|:---:|:---|
| **Eigen3** | >= 3.1.0 | Linear algebra library |
| **OpenCV** | >= 3.4.0 | Image processing |
| **Pangolin** | Latest | Visualization and GUI |
| **ONNX Runtime** | >= 1.14.0 | Deep learning inference |
| **PCL** | >= 1.10.0 | Point cloud processing |
| **Boost** | >= 1.58.0 | Serialization support |
| **g2o** | Bundled | Graph optimization |
| **DBoW2** | Bundled | Loop detection |
| **Sophus** | Bundled | Lie group operations |

### Optional Dependencies

- **ROS** (Noetic/Foxy): For processing bag files
- **TensorRT**: For NVIDIA GPU acceleration
- **CUDA**: Required for TensorRT

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/TP-DSLAM.git
cd TP-DSLAM
```

### 2. Install Dependencies

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y cmake libeigen3-dev libopencv-dev libboost-all-dev

# Install Pangolin
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..

# Install ONNX Runtime (GPU version)
# Download from: https://github.com/microsoft/onnxruntime/releases
# Extract to ./onnxruntime_gpu/
```

### 3. Build the Project

```bash
chmod +x build.sh
./build.sh
```

### 4. Download Pre-trained Models

Download the YOLO segmentation model and place it in the project root:

```bash
# The model file yolo11n-seg.onnx should be in the project root directory
# Download from: [Provide download link]
```

---

## Running the System

### RGB-D Mode (TUM Dataset)

```bash
./Examples/RGB-D/rgbd_tum \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/RGB-D/TUM3.yaml \
    /path/to/TUM_dataset/rgbd_dataset_freiburg3_walking_xyz \
    ./Examples/RGB-D/TUM3.txt
```

### Stereo Mode (KITTI Dataset)

```bash
./Examples/Stereo/stereo_kitti \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/Stereo/KITTI00-02.yaml \
    /path/to/kitti/sequences/00/
```

### Stereo-Inertial Mode (EuRoC Dataset)

```bash
./Examples/Stereo-Inertial/stereo_inertial_euroc \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/Stereo-Inertial/EuRoC.yaml \
    /path/to/euroc/mav0/
```

### Monocular Mode

```bash
./Examples/Monocular/mono_tum \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/Monocular/TUM1.yaml \
    /path/to/TUM_dataset/
```

### With RealSense Camera

```bash
# RGB-D with RealSense D435i
./Examples/RGB-D/rgbd_realsense_D435i \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/RGB-D/RealSense_D435i.yaml
```

---

## Configuration

### YAML Configuration Files

Each sensor mode has pre-configured YAML files for different datasets:

| Dataset | Configuration File | Description |
|:---|:---|:---|
| TUM RGB-D | `TUM1.yaml`, `TUM2.yaml`, `TUM3.yaml` | Different camera calibrations |
| KITTI | `KITTI00-02.yaml`, `KITTI03.yaml`, `KITTI04-12.yaml` | Stereo sequences |
| EuRoC | `EuRoC.yaml` | MAV sequences |
| VIODE | `VIODE.yaml` | Dynamic outdoor sequences |

### Key Configuration Parameters

```yaml
# Dynamic Detection Parameters
DynamicDetector.modelPath: "yolo11n-seg.onnx"
DynamicDetector.confThres: 0.4
DynamicDetector.scoreThres: 0.25
DynamicDetector.nmsThres: 0.45

# ORB Parameters
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8

# Camera Parameters
Camera.width: 640
Camera.height: 480
Camera.fps: 30
```

---

## Project Structure

```
TP-DSLAM/
├── include/                    # Header files
│   ├── CameraModels/          # Camera model implementations
│   ├── DynamicDetector.h      # Dynamic object detection
│   ├── Frame.h                # Frame processing
│   ├── KeyFrame.h             # Keyframe management
│   ├── Map.h                  # Map structure
│   ├── Optimizer.h            # Graph optimization
│   ├── System.h               # Main system interface
│   ├── Tracking.h             # Tracking module
│   └── ...
├── src/                       # Source files
│   ├── DynamicDetector.cc     # YOLO-based detection
│   ├── Frame.cc               # Frame processing
│   ├── Tracking.cc            # Tracking implementation
│   └── ...
├── Examples/                  # Example executables
│   ├── Monocular/             # Monocular examples
│   ├── Stereo/                # Stereo examples
│   ├── RGB-D/                 # RGB-D examples
│   ├── Monocular-Inertial/    # Monocular + IMU
│   ├── Stereo-Inertial/       # Stereo + IMU
│   └── RGB-D-Inertial/        # RGB-D + IMU
├── Thirdparty/                # Third-party libraries
│   ├── DBoW2/                 # Bag of Words
│   ├── g2o/                   # Graph optimization
│   └── Sophus/                # Lie groups
├── Vocabulary/                # ORB vocabulary
├── onnxruntime_gpu/           # ONNX Runtime libraries
├── scripts/                   # Build scripts
├── evaluation/                # Evaluation tools
├── CMakeLists.txt             # Main CMake file
├── build.sh                   # Build script
└── README.md                  # This file
```

---

## Evaluation Tools

### Absolute Trajectory Error (ATE)

```bash
# Generate trajectory file
./Examples/RGB-D/rgbd_tum ...  # Produces CameraTrajectory.txt

# Evaluate ATE
python evaluation/evaluate_ate_scale.py \
    evaluation/Ground_truth/TUM/fr3_walking_xyz.txt \
    CameraTrajectory.txt \
    --plot results.png
```

### Association Script

```bash
# Associate RGB and depth images
python evaluation/associate.py \
    /path/to/rgb.txt \
    /path/to/depth.txt \
    > associations.txt
```

---

## Advanced Features

### TensorRT Acceleration (Jetson)

For NVIDIA Jetson devices, enable TensorRT acceleration:

```bash
# Build with TensorRT support
./scripts/build_jetson.sh

# Or manually
mkdir build && cd build
cmake .. -DENABLE_TENSORRT=ON -DTENSORRT_ROOT=/usr/lib/aarch64-linux-gnu
make -j$(nproc)
```

### ONNX to TensorRT Conversion

```bash
# Convert ONNX model to TensorRT engine
./scripts/convert_onnx_to_tensorrt.sh yolo11n-seg.onnx yolo11n-seg.engine
```

<!-- ---

## Citation

If you find TP-DSLAM useful in your research, please consider citing our paper:

```bibtex
@article{TP-DSLAM2025,
  title={TP-DSLAM: Robust Dynamic SLAM with Temporal-Semantic Fusion and Geometric Gating},
  author={[Author Names]},
  journal={[Journal/Conference]},
  year={2025}
}
``` -->

---

## Acknowledgments

TP-DSLAM is built upon the excellent work of:

- **ORB-SLAM3**: [Carlos Campos et al., IEEE TPAMI 2021]
- **DBoW2**: [Dorian Gálvez-López et al.]
- **g2o**: [Rainer Kümmerle et al.]
- **Sophus**: [Hauke Strasdat]
- **ONNX Runtime**: [Microsoft]
- **YOLO**: [Ultralytics]

We thank the authors for making their code publicly available.

---

## License

TP-DSLAM is released under the [GNU General Public License v3.0](LICENSE).

This means you are free to use, modify, and distribute this software, provided that any derivative works are also released under the same license. See the [LICENSE](LICENSE) file for full details.

---

## Contact

For questions, issues, or collaborations, please:

- Open an issue on [GitHub](https://github.com/your-username/TP-DSLAM/issues)
- Contact the authors at: [your-email@example.com]

---

## Star History

If you find this project helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/TP-DSLAM&type=Date)](https://star-history.com/#your-username/TP-DSLAM&Date)

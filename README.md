# TP-DSLAM: Robust Dynamic SLAM with Temporal-Semantic Fusion and Geometric Gating

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/platform-Ubuntu_22.04-orange.svg)](https://ubuntu.com/)
[![Language](https://img.shields.io/badge/language-C++-blue.svg)](https://isocpp.org/)

**TP-DSLAM** is a robust visual SLAM system designed for dynamic environments. By integrating **Temporal-Semantic Fusion** and a **Geometric Disparity Gating** mechanism, our system effectively mitigates the interference of dynamic objects (e.g., walking persons, moving vehicles) and maintains high-precision localization even under challenging conditions like fast motion and lighting variations.

> **Note:** This repository contains the source code for the paper *[Insert Paper Title Here]*.

## 🌟 Key Features

-   **Dynamic Object Robustness:** Utilizes a deep learning-based semantic segmentation module combined with a **Time-Semantic Cache** to handle occlusions and temporary dynamic noise.
-   **Geometric Consistency:** Implements a **Disparity Gating** mechanism to filter outliers based on geometric constraints, significantly improving tracking stability.
-   **Dynamic Confidence Weighting:** Unlike hard-rejection methods, we employ a soft-weighting strategy to preserve useful features while suppressing dynamic noise.
-   **Versatility:** Supports Monocular, Stereo, and RGB-D modes.
-   **Real-time Performance:** Optimized to run efficiently on standard laptop hardware (tested on 13th Gen Intel Core i5).

## 📊 Performance Evaluation

We evaluated TP-DSLAM on standard benchmarks including **TUM RGB-D**, **KITTI**, **EuRoC**, and **VIODE**. Our system demonstrates state-of-the-art performance, particularly in highly dynamic scenarios.

### Absolute Trajectory Error (RMSE) on TUM RGB-D (fr3/walking sequences)

| Sequence | ORB-SLAM3 | DynaSLAM | YDD-SLAM | **TP-DSLAM (Ours)** |
| :--- | :---: | :---: | :---: | :---: |
| **w_half** | 0.312 | 0.025 | 0.028 | **0.026** |
| **w_xyz** | 0.510 | **0.015** | **0.015** | **0.015** |
| **w_rpy** | 0.536 | **0.035** | 0.036 | **0.035** |
| **Avg. (Dyn)** | 0.381 | 0.020 | 0.022 | **0.021** |

### Robustness in Challenging Conditions (EuRoC & VIODE)

TP-DSLAM significantly outperforms baselines in sequences with fast motion and low-light conditions.

-   **EuRoC (V203):** Reduced RMSE from **0.233m** (ORB-SLAM3) to **0.077m** (Ours).
-   **VIODE (City_night_High):** Maintained stability where ORB-SLAM3 failed (Error > 2.7m), achieving an RMSE of **0.477m**.

## 🛠️ Prerequisites

We have tested the system on **Ubuntu 20.04**. Ensure you have the following dependencies installed:

-   **C++ Compiler:** GCC >= 7.5 (Support for C++11/14)
-   **CMake:** >= 3.10
-   **Eigen3:** >= 3.1.0
-   **OpenCV:** >= 3.4.0 (Required for image processing)
-   **Pangolin:** For visualization and GUI
-   **PyTorch & LibTorch:** Required for the semantic segmentation module (Dynamic object detection)
-   **ROS (Optional):** Recommended for processing bag files (ROS1 Noetic or ROS2 Foxy)

## 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/TP-DSLAM.git
    cd TP-DSLAM
    ```

2.  **Build the project:**
    ```bash
    chmod +x build.sh
    ./build.sh
    ```

3.  **Download Pre-trained Models:**
    Please download the semantic segmentation model weights and place them in the `models/` directory.

## 🏃‍♂️ Running the System

### 1. Monocular/RGB-D Example (TUM Dataset)
To run the system on a TUM sequence (e.g., `fr3_walking_xyz`):

```bash
./bin/rgbd_tum \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/RGB-D/TUM3.yaml \
    /path/to/TUM_dataset/rgbd_dataset_freiburg3_walking_xyz \
    ./Examples/RGB-D/TUM3.txt
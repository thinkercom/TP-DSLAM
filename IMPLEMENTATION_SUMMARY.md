# TP-DSLAM Core Innovation Implementation Summary

## Overview

This document summarizes the implementation of the four core innovations described in the TP-DSLAM paper. The changes were made to enable **Temporal-Semantic Fusion (TLSP)**, **Disparity-Gated Confidence Modulation (DGCM)**, and **Dynamic Adaptive Pose Optimization**.

---

## 1. TLSP: Temporal-Semantic Propagation (Core Innovation 1)

### Problem
Original code used simple cache copying for non-inference frames, without considering camera motion or object motion.

### Solution Implemented

#### 1.1 Pose-Guided Semantic Warping (`Tracking.cc`)

Added `WarpSemanticMap()` function that:
- Uses camera pose transformation to project semantic maps from source frame to destination frame
- Back-projects 2D pixels to 3D using depth information
- Transforms 3D points using relative pose: `T_dst_src = T_dst_w * T_w_src`
- Projects transformed points to destination image plane
- Uses max pooling for conservative semantic propagation

```cpp
cv::Mat Tracking::WarpSemanticMap(const cv::Mat &srcSemantic, const Sophus::SE3f &srcPose, 
                                  const Sophus::SE3f &dstPose, const cv::Mat &depthMap)
```

#### 1.2 Modified `ProcessDynamicPrior()`

- **Inference frames**: Run YOLO, cache result with current pose
- **Non-inference frames**: Warp cached semantic map using pose-guided projection
- Falls back to direct cache copy if depth information is insufficient

#### 1.3 New Member Variables (`Tracking.h`)

```cpp
cv::Mat mLastSemanticMap;           // Semantic map from last frame
Sophus::SE3f mLastSemanticPose;     // Pose when semantic map was computed
bool mbHasLastSemantic;             // Flag indicating if we have valid cached semantic
```

### Files Modified
- `include/Tracking.h` - Added member variables and function declarations
- `src/Tracking.cc` - Implemented `WarpSemanticMap()` and modified `ProcessDynamicPrior()`

---

## 2. DGCM: Disparity-Gated Confidence Modulation (Core Innovation 2)

### Problem
Original code used simple `(1 - dyn_prob) * geo_score` fusion without considering stereo disparity characteristics.

### Solution Implemented

#### 2.1 Sigmoid Disparity Gate Function (`Tracking.cc`)

```cpp
float Tracking::SigmoidDisparityGate(float disparity, float d0, float k)
{
    // g(d) = 1 / (1 + exp(-k*(d - d0)))
    // Near field (large disparity): g -> 1, rely more on geometry
    // Far field (small disparity): g -> 0, rely more on semantics
    return 1.0f / (1.0f + std::exp(-k * (disparity - d0)));
}
```

#### 2.2 `ComputeDisparityGatedConfidence()` Function

For each keypoint:
1. **Get semantic dynamic prior** from YOLO detection
2. **Compute disparity** from stereo/RGB-D data
3. **Apply sigmoid gating**:
   - Near field (disparity > threshold): `geo_weight → 1` (geometry reliable)
   - Far field (disparity < threshold): `geo_weight → 0` (semantics more reliable)
4. **Fuse scores**:
   ```
   fused_score = geo_weight * geo_score + (1 - geo_weight) * 1.0
   reliability = (1 - semantic_prior) * fused_score
   ```

#### 2.3 Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `disparity_threshold_near` | 15.0f | Near/far field boundary (pixels) |
| `disparity_gate_k` | 0.3f | Sigmoid steepness |

### Files Modified
- `include/Tracking.h` - Added function declarations
- `src/Tracking.cc` - Implemented DGCM functions
- Modified `GrabImageStereo()`, `GrabImageRGBD()`, `GrabImageMonocular()` to call DGCM

---

## 3. Dynamic Adaptive Pose Optimization (Core Innovation 3)

### Problem
Original optimizer used fixed Huber kernel thresholds regardless of dynamic content.

### Solution Implemented

#### 3.1 Dynamic Ratio Computation

```cpp
float Tracking::ComputeDynamicRatio(const ORB_SLAM3::Frame &F, float threshold)
{
    // Count features with dynamic prior > threshold
    // Return ratio of dynamic features to total features
}
```

#### 3.2 Adaptive Huber Delta Computation

```cpp
float Tracking::ComputeAdaptiveHuberDelta(float dynamicRatio, float baseDelta, 
                                           float minDelta, float maxDelta)
{
    // Higher dynamic ratio -> more conservative (smaller delta)
    // Lower dynamic ratio -> more relaxed (larger delta)
    float adaptiveDelta = baseDelta * (1.0f - dynamicRatio) + minDelta * dynamicRatio;
    return std::max(minDelta, std::min(maxDelta, adaptiveDelta));
}
```

#### 3.3 Modified `LocalBundleAdjustment()` in `Optimizer.cc`

**Key Changes:**

1. **Compute dynamic ratio from local keyframes**:
   ```cpp
   float dynamicRatio = static_cast<float>(dynamicCount) / totalCount;
   ```

2. **Adaptive threshold scaling**:
   ```cpp
   float adaptiveFactor = 1.0f - 0.7f * dynamicRatio;
   const float thHuberMono = thHuberMonoBase * adaptiveFactor;
   const float thHuberStereo = thHuberStereoBase * adaptiveFactor;
   ```

3. **Per-point reliability weighting**:
   ```cpp
   // Weight information matrix by static reliability
   e->setInformation(Eigen::Matrix2d::Identity() * invSigma2 * staticReliability);
   
   // Further reduce delta for low reliability points
   float pointDelta = thHuberMono * (0.5f + 0.5f * staticReliability);
   rk->setDelta(pointDelta);
   ```

### Files Modified
- `include/Tracking.h` - Added adaptive computation functions
- `src/Tracking.cc` - Implemented adaptive functions
- `src/Optimizer.cc` - Modified `LocalBundleAdjustment()` with adaptive Huber kernel

---

## 4. KITTI Highway Scenario Optimization (New)

### Problem
KITTI 01 sequence (highway scenario) has very poor ATE (~14m) due to:
- High-speed moving vehicles (same/opposite direction)
- Feature degradation (guardrails, vegetation, sky)
- Motion blur and long-range features with small parallax
- Lack of geometric structures

### Solution Implemented

#### 4.1 Increased Vehicle Dynamic Prior (`DynamicDetector.cc`)

```cpp
// Before: Car/Bus/Truck = 0.70f
// After:  Car/Bus/Truck = 0.85f (highway vehicles are usually moving)
{2, 0.85f},  // car
{5, 0.85f},  // bus
{7, 0.85f},  // truck
```

#### 4.2 Configurable Dynamic Detection Parameters (`Tracking.h`, `Tracking.cc`)

Added new member variables:
```cpp
float mDynamicDropThreshold;   // Configurable threshold (default: 0.6)
int mDynamicSkipFrames;        // Configurable skip frames (default: 2)
bool mHighwayMode;             // Highway mode flag
```

Parameters are read from YAML config file:
```yaml
DynamicDetector.dropThreshold: 0.45
DynamicDetector.skipFrames: 1
DynamicDetector.highwayMode: 1
```

#### 4.3 Highway Mode Logic

When `highwayMode = 1`:
- Threshold reduced to `min(config_value, 0.45f)` (more aggressive filtering)
- Skip frames reduced by 1 (more responsive to fast-moving objects)

#### 4.4 Updated KITTI Config (`Examples/Stereo/KITTI00-02.yaml`)

```yaml
# Increased features for highway
ORBextractor.nFeatures: 3000  # was 2000

# Highway-specific dynamic detection
DynamicDetector.dropThreshold: 0.45
DynamicDetector.skipFrames: 1
DynamicDetector.highwayMode: 1
```

### Files Modified
- `src/DynamicDetector.cc` - Increased vehicle dynamic priors
- `include/Tracking.h` - Added configurable parameters
- `src/Tracking.cc` - Read parameters from config, highway mode logic
- `Examples/Stereo/KITTI00-02.yaml` - Added highway mode config

---

## 5. Lightweight Embedded Architecture (Core Innovation 4)

### Status: Already Implemented (Minor Enhancements)

The original codebase already had:
- ✅ ORB-SLAM3 tracking framework modification
- ✅ Modular dynamic detection (DynamicDetector class)
- ✅ TensorRT acceleration support (experimental)
- ✅ Frame skipping inference (SKIP_FRAMES = 2)

### No Major Changes Required

---

## Summary of All Modified Files

| File | Changes |
|------|---------|
| `include/Tracking.h` | Added TLSP member variables, DGCM function declarations, adaptive optimization functions, configurable dynamic detection parameters |
| `include/KeyFrame.h` | Added `mvDynPrior`, `mvGeoScore`, `mvStaticReliability` member variables |
| `src/Tracking.cc` | Implemented TLSP warping, DGCM modulation, adaptive computations, modified all GrabImage functions, added highway mode support |
| `src/KeyFrame.cc` | Initialize dynamic confidence member variables |
| `src/Optimizer.cc` | Modified `LocalBundleAdjustment()` with adaptive Huber kernel and reliability weighting |
| `src/DynamicDetector.cc` | Increased vehicle dynamic priors for highway scenarios |
| `Examples/Stereo/KITTI00-02.yaml` | Added highway mode configuration parameters |

---

## Build Instructions

```bash
cd /home/cmt/Projects/1-TP-DSLAM/TP-DSLAM
./build.sh
```

---

## Testing Recommendations

### 1. TUM RGB-D Dataset (dynamic sequences)
- `fr3/walking_xyz`
- `fr3/walking_halfsphere`
- `fr3/walking_rpy`

### 2. KITTI Dataset (highway scenarios)
- **KITTI 01** - Highway (most challenging, use highway mode)
- **KITTI 00, 02** - Urban (standard mode)
- **KITTI 03-12** - Various scenarios

### Running with Highway Mode
```bash
# KITTI 01 with highway mode (optimized config)
./Examples/Stereo/stereo_kitti \
    ./Vocabulary/ORBvoc.txt \
    ./Examples/Stereo/KITTI00-02.yaml \
    /path/to/kitti/sequences/01/
```

### Key Metrics to Monitor
- ATE RMSE (should decrease significantly for highway)
- Tracking success rate
- Dynamic feature rejection rate
- Processing time per frame

### Debug Output
- `[TP-DSLAM]` prefix for new debug messages
- `[TP-DSLAM] Highway Mode Enabled:` - confirms highway mode active
- Dynamic ratio statistics every 100 frames
- Adaptive Huber delta values in verbose mode

### Expected Results
| Dataset | ORB-SLAM3 | TP-DSLAM (Standard) | TP-DSLAM (Highway) |
|---------|-----------|---------------------|-------------------|
| KITTI 01 | ~14m | ~10m | ~5-7m (target) |
| TUM fr3/walking | ~0.31m | ~0.03m | - |

---

## Future Improvements

1. **TLSP Enhancement**:
   - Implement optical flow-based warping for better boundary handling
   - Add uncertainty estimation for warped regions

2. **DGCM Enhancement**:
   - Learn disparity threshold from data
   - Add temporal consistency for confidence scores

3. **Adaptive Optimization**:
   - Implement full SE(3) manifold optimization with confidence weights
   - Add outlier rejection based on dynamic probability

4. **Performance**:
   - Implement async YOLO inference pipeline
   - Add INT8 quantization for TensorRT

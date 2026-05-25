cmake_minimum_required(VERSION 3.5)
project(TP-DSLAM)

# ==========================================
# 1. Python3 and NumPy configuration(for Conda)
# ==========================================

# if not define Python3_ROOT_DIR
# try to use the activated Conda environment
if(DEFINED ENV{CONDA_PREFIX})
    set(CONDA_ENV_PATH $ENV{CONDA_PREFIX})
    message(STATUS "Detected Conda Environment: ${CONDA_ENV_PATH}")
    
    # use python from Conda environment
    set(Python3_ROOT_DIR "${CONDA_ENV_PATH}")
    set(Python3_FIND_STRATEGY "LOCATION")
endif()

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

message(STATUS "Using Python Executable: ${Python3_EXECUTABLE}")
message(STATUS "Python Include Dirs: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python Libraries: ${Python3_LIBRARIES}")

# get NumPy include path
if(Python3_Interpreter_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NUMPY_INCLUDE_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE NUMPY_RESULT
    )
    
    if(NOT NUMPY_RESULT EQUAL 0)
        message(FATAL_ERROR "NumPy is not installed in the selected Python environment: ${Python3_EXECUTABLE}")
    endif()
    message(STATUS "NumPy Include Path: ${NUMPY_INCLUDE_PATH}")
else()
    message(FATAL_ERROR "Python3 Interpreter not found!")
endif()

include_directories(
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_PATH}
)

# ==========================================
# 2. basic config setting
# ==========================================

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wno-array-bounds  -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
add_definitions(-DCOMPILEDWITHC14)
message(STATUS "Using flag -std=c++14 to support Pangolin/sigslot.")

# add conda environment to CMAKE_PREFIX_PATH
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# ==========================================
# 3. find dependencies
# ==========================================

set(ONNXRUNTIME_ROOT "./onnxruntime_gpu") # your onnxruntime_gpu path
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(realsense2)
find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# print version
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 Include: ${EIGEN3_INCLUDE_DIR}")

# ======================== experiment ========================
# [EXPERIMENT] TensorRT support for Jetson Orin NX
# Purpose: Enable TensorRT acceleration for YOLO inference
option(ENABLE_TENSORRT "Enable TensorRT acceleration" OFF) # you can open this module to enable TensorRT

if(ENABLE_TENSORRT)
    message(STATUS "TensorRT acceleration enabled")
    
    # Find TensorRT
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS /usr/include/x86_64-linux-gnu
              /usr/include/aarch64-linux-gnu
              /usr/include
              ${TENSORRT_ROOT}/include
    )
    
    find_library(TENSORRT_LIBRARY nvinfer
        HINTS /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu
              /usr/lib
              ${TENSORRT_ROOT}/lib
    )
    
    find_library(NVONNXPARSER_LIBRARY nvonnxparser
        HINTS /usr/lib/x86_64-linux-gnu
              /usr/lib/aarch64-linux-gnu
              /usr/lib
              ${TENSORRT_ROOT}/lib
    )
    
    if(TENSORRT_INCLUDE_DIR AND TENSORRT_LIBRARY)
        message(STATUS "TensorRT found: ${TENSORRT_INCLUDE_DIR}")
        include_directories(${TENSORRT_INCLUDE_DIR})
        add_definitions(-DENABLE_TENSORRT)
    else()
        message(WARNING "TensorRT not found, disabling TensorRT support")
        set(ENABLE_TENSORRT OFF)
    endif()
    
    # Find CUDA
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
endif()
# ======================== experiment end =====================

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/include/CameraModels
    ${PROJECT_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${Pangolin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# ==========================================
# 4. build library
# ==========================================

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Atlas.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Viewer.cc
src/ImuTypes.cc
src/G2oTypes.cc
src/CameraModels/Pinhole.cpp
src/CameraModels/KannalaBrandt8.cpp
src/OptimizableTypes.cpp
src/MLPnPsolver.cpp
src/GeometricTools.cc
src/TwoViewReconstruction.cc
src/Config.cc
src/Settings.cc
src/DynamicDetector.cc
include/System.h
include/Tracking.h
include/LocalMapping.h
include/LoopClosing.h
include/ORBextractor.h
include/ORBmatcher.h
include/FrameDrawer.h
include/Converter.h
include/MapPoint.h
include/KeyFrame.h
include/Atlas.h
include/Map.h
include/MapDrawer.h
include/Optimizer.h
include/Frame.h
include/KeyFrameDatabase.h
include/Sim3Solver.h
include/Viewer.h
include/ImuTypes.h
include/G2oTypes.h
include/CameraModels/GeometricCamera.h
include/CameraModels/Pinhole.h
include/CameraModels/KannalaBrandt8.h
include/OptimizableTypes.h
include/MLPnPsolver.h
include/GeometricTools.h
include/TwoViewReconstruction.h
include/SerializationUtils.h
include/Config.h
include/Settings.h
include/DynamicDetector.h)

add_subdirectory(Thirdparty/g2o)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
${Python3_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
-lboost_serialization
-lcrypto

onnxruntime # added
${PCL_LIBRARIES}
)

# ======================== experiment ========================
# [EXPERIMENT] Link TensorRT libraries
if(ENABLE_TENSORRT)
    target_link_libraries(${PROJECT_NAME}
        ${TENSORRT_LIBRARY}
        ${NVONNXPARSER_LIBRARY}
        ${CUDA_LIBRARIES}
        nvinfer_plugin
    )
endif()
# ======================== experiment end =====================

target_link_libraries(${PROJECT_NAME} -lpthread)

# If RealSense SDK is found the library is added and its examples compiled
if(realsense2_FOUND)
    include_directories(${PROJECT_NAME}
    ${realsense2_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME}
    ${realsense2_LIBRARY}
    )
endif()


# ==========================================
# 5. build example programs
# ==========================================

# RGB-D examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
        Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(rgbd_realsense_D435i
            Examples/RGB-D/rgbd_realsense_D435i.cc)
    target_link_libraries(rgbd_realsense_D435i ${PROJECT_NAME})
endif()


# RGB-D inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D-Inertial)

if(realsense2_FOUND)
    add_executable(rgbd_inertial_realsense_D435i
            Examples/RGB-D-Inertial/rgbd_inertial_realsense_D435i.cc)
    target_link_libraries(rgbd_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
        Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})

add_executable(stereo_tum_vi
        Examples/Stereo/stereo_tum_vi.cc)
target_link_libraries(stereo_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_realsense_t265
            Examples/Stereo/stereo_realsense_t265.cc)
    target_link_libraries(stereo_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_realsense_D435i
            Examples/Stereo/stereo_realsense_D435i.cc)
    target_link_libraries(stereo_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
        Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
        Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
        Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

add_executable(mono_tum_vi
        Examples/Monocular/mono_tum_vi.cc)
target_link_libraries(mono_tum_vi ${PROJECT_NAME})

add_executable(mono_eth3d
        Examples/Monocular/mono_eth3d.cc)
target_link_libraries(mono_eth3d ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_realsense_t265
            Examples/Monocular/mono_realsense_t265.cc)
    target_link_libraries(mono_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_realsense_D435i
            Examples/Monocular/mono_realsense_D435i.cc)
    target_link_libraries(mono_realsense_D435i ${PROJECT_NAME})
endif()

#Monocular inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular-Inertial)

add_executable(mono_inertial_euroc
        Examples/Monocular-Inertial/mono_inertial_euroc.cc)
target_link_libraries(mono_inertial_euroc ${PROJECT_NAME})

add_executable(mono_inertial_tum_vi
        Examples/Monocular-Inertial/mono_inertial_tum_vi.cc)
target_link_libraries(mono_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(mono_inertial_realsense_t265
            Examples/Monocular-Inertial/mono_inertial_realsense_t265.cc)
    target_link_libraries(mono_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(mono_inertial_realsense_D435i
            Examples/Monocular-Inertial/mono_inertial_realsense_D435i.cc)
    target_link_libraries(mono_inertial_realsense_D435i ${PROJECT_NAME})
endif()

#Stereo Inertial examples
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo-Inertial)

add_executable(stereo_inertial_euroc
        Examples/Stereo-Inertial/stereo_inertial_euroc.cc)
target_link_libraries(stereo_inertial_euroc ${PROJECT_NAME})

add_executable(stereo_inertial_tum_vi
        Examples/Stereo-Inertial/stereo_inertial_tum_vi.cc)
target_link_libraries(stereo_inertial_tum_vi ${PROJECT_NAME})

if(realsense2_FOUND)
    add_executable(stereo_inertial_realsense_t265
            Examples/Stereo-Inertial/stereo_inertial_realsense_t265.cc)
    target_link_libraries(stereo_inertial_realsense_t265 ${PROJECT_NAME})

    add_executable(stereo_inertial_realsense_D435i
            Examples/Stereo-Inertial/stereo_inertial_realsense_D435i.cc)
    target_link_libraries(stereo_inertial_realsense_D435i ${PROJECT_NAME})
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Calibration)
if(realsense2_FOUND)
    add_executable(recorder_realsense_D435i
            Examples/Calibration/recorder_realsense_D435i.cc)
    target_link_libraries(recorder_realsense_D435i ${PROJECT_NAME})

    add_executable(recorder_realsense_T265
            Examples/Calibration/recorder_realsense_T265.cc)
    target_link_libraries(recorder_realsense_T265 ${PROJECT_NAME})
endif()
/**
 * This file is part of ORB-SLAM3
 *
 * Copyright (C) 2017-2021 Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 * Copyright (C) 2014-2016 Raúl Mur-Artal, José M.M. Montiel and Juan D. Tardós, University of Zaragoza.
 *
 * ORB-SLAM3 is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM3 is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with ORB-SLAM3.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>

#include <System.h>

using namespace std;

// 函数声明：加载 ETH3D 格式数据 (rgb.txt)
void LoadImages(const string &strPathToSequence,
                vector<string> &vstrImages, vector<double> &vTimeStamps);

int main(int argc, char **argv)
{
    // 修改参数检查：只需要 3 个参数 (vocab, settings, sequence_path)
    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./mono_eth3d path_to_vocabulary path_to_settings path_to_sequence" << endl;
        cerr << "Example: ./mono_eth3d Vocabulary/ORBvoc.txt ETH3D_Mono.yaml ETH3D/camera_shake_1" << endl;
        return 1;
    }

    // 加载图像和时间戳
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;

    cout << "Loading images from: " << string(argv[3]) << "...";
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);
    cout << " LOADED!" << endl;

    int nImages = vstrImageFilenames.size();
    if (nImages == 0)
    {
        cerr << "No images found!" << endl;
        return 1;
    }

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl
         << "-------" << endl;
    cout.precision(17);

    int fps = 20; // ETH3D standard FPS
    float dT = 1.f / fps;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // 第三个参数 MONOCULAR, 第四个参数 false (不启用可视化窗口中的 IMU 信息，虽然单目也没 IMU)
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::MONOCULAR, true);
    float imageScale = SLAM.GetImageScale();

    double t_resize = 0.f;
    double t_track = 0.f;

    // Main loop
    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {
        // Read image from file
        im = cv::imread(vstrImageFilenames[ni], cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        // Resize if necessary (based on YAML Camera.newWidth/newHeight)
        if (imageScale != 1.f)
        {
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::steady_clock::time_point t_Start_Resize = std::chrono::steady_clock::now();
#endif
#endif
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
#ifdef REGISTER_TIMES
#ifdef COMPILEDWITHC11
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
#else
            std::chrono::steady_clock::time_point t_End_Resize = std::chrono::steady_clock::now();
#endif
            t_resize = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t_End_Resize - t_Start_Resize).count();
            SLAM.InsertResizeTime(t_resize);
#endif
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#endif

        // Pass the image to the SLAM system
        // 注意：TrackMonocular 的参数顺序 (image, timestamp, imu_points, frame_name)
        // 这里没有 IMU 数据，传入空 vector
        SLAM.TrackMonocular(im, tframe, vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenames[ni]);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#endif

#ifdef REGISTER_TIMES
        t_track = t_resize + std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t2 - t1).count();
        SLAM.InsertTrackTime(t_track);
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame (Real-time simulation)
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
        {
            usleep((T - ttrack) * 1e6);
        }
    }

    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory (EuRoC format: tx ty tz qx qy qz qw)
    // 默认保存为 CameraTrajectory.txt 和 KeyFrameTrajectory.txt
    SLAM.SaveTrajectoryEuRoC("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC("KeyFrameTrajectory.txt");

    cout << endl
         << "-------" << endl;
    cout << "Median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "Mean tracking time: " << accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0) / nImages << endl;
    cout << "Trajectory saved to CameraTrajectory.txt and KeyFrameTrajectory.txt" << endl;

    return 0;
}

// 加载 ETH3D 数据的核心函数
void LoadImages(const string &strPathToSequence,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    // ETH3D 的时间戳文件是 rgb.txt
    string strPathTimes = strPathToSequence + "/rgb.txt";

    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());

    if (!fTimes.is_open())
    {
        cerr << "Failed to open times file: " << strPathTimes << endl;
        exit(1);
    }

    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);

    string line;
    while (getline(fTimes, line))
    {
        if (line.empty() || line[0] == '#')
            continue; // 跳过空行和注释

        stringstream ss;
        ss << line;

        double t;
        string filename;

        // 读取：时间戳 文件名
        // 例如：1465479000.123456 rgb/000000.png
        if (ss >> t >> filename)
        {
            vTimeStamps.push_back(t); // ETH3D 时间戳单位已经是秒，不需要 *1e-9

            // 构建完整路径：SequencePath + "/" + Filename
            // 注意：rgb.txt 中的 filename 通常是相对路径 "rgb/000000.png"
            vstrImages.push_back(strPathToSequence + "/" + filename);
        }
    }

    fTimes.close();
}
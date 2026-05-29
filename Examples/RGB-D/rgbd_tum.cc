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

#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include<numeric>

#include<opencv2/core/core.hpp>

#include<System.h>

#ifdef __linux__
#include <sys/resource.h>
#include <unistd.h>
#endif

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::RGBD,true);
    float imageScale = SLAM.GetImageScale();

    // Vector for tracking time statistics (processing time only)
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    for(int ni=0; ni<nImages; ni++)
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],cv::IMREAD_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],cv::IMREAD_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        // Track RGBD frame (pure SLAM processing time)
        auto t1 = chrono::steady_clock::now();
        SLAM.TrackRGBD(imRGB,imD,tframe);
        auto t2 = chrono::steady_clock::now();

        // Calculate processing time
        double ttrack = chrono::duration_cast<chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // Wait to simulate real-time operation
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // ======================== Final Performance Statistics ========================
    cout << endl;
    cout << "====================================================================" << endl;
    cout << "                    PERFORMANCE STATISTICS                          " << endl;
    cout << "====================================================================" << endl;

    // Sort for percentile calculations
    vector<float> vTimesTrackSorted = vTimesTrack;
    sort(vTimesTrackSorted.begin(), vTimesTrackSorted.end());

    // Calculate statistics
    float sum_track = accumulate(vTimesTrack.begin(), vTimesTrack.end(), 0.0f);
    float mean_track = sum_track / nImages;
    float median_track = vTimesTrackSorted[nImages/2];
    float min_track = vTimesTrackSorted.front();
    float max_track = vTimesTrackSorted.back();
    float std_track = 0.0f;
    for(int i = 0; i < nImages; i++)
        std_track += (vTimesTrack[i] - mean_track) * (vTimesTrack[i] - mean_track);
    std_track = sqrt(std_track / nImages);

    // Percentiles
    float p95_track = vTimesTrackSorted[(int)(nImages * 0.95)];
    float p99_track = vTimesTrackSorted[(int)(nImages * 0.99)];

    // Processing FPS (algorithm actual speed)
    float processing_fps_mean = 1.0f / mean_track;
    float processing_fps_median = 1.0f / median_track;

    // System FPS (based on dataset timestamps)
    float total_dataset_time = vTimestamps.back() - vTimestamps.front();
    float system_fps = (nImages - 1) / total_dataset_time;

    cout << fixed << setprecision(2);
    cout << endl;
    cout << "--- Processing Time (SLAM Algorithm Only) ---" << endl;
    cout << "  Mean:     " << mean_track * 1000.0f << " ms" << endl;
    cout << "  Median:   " << median_track * 1000.0f << " ms" << endl;
    cout << "  Std Dev:  " << std_track * 1000.0f << " ms" << endl;
    cout << "  Min:      " << min_track * 1000.0f << " ms" << endl;
    cout << "  Max:      " << max_track * 1000.0f << " ms" << endl;
    cout << "  P95:      " << p95_track * 1000.0f << " ms" << endl;
    cout << "  P99:      " << p99_track * 1000.0f << " ms" << endl;

    cout << endl;
    cout << "--- FPS Statistics ---" << endl;
    cout << "  Processing FPS (Mean):   " << processing_fps_mean << " fps" << endl;
    cout << "  Processing FPS (Median): " << processing_fps_median << " fps" << endl;
    cout << "  System FPS (Dataset):    " << system_fps << " fps" << endl;

    cout << endl;
    cout << "--- Dataset Information ---" << endl;
    cout << "  Total Frames:       " << nImages << endl;
    cout << "  Dataset Duration:   " << total_dataset_time << " s" << endl;
    cout << "  Dataset Frame Rate: " << system_fps << " fps" << endl;

    cout << endl;
    cout << "--- Real-time Capability ---" << endl;
    if(mean_track < (1.0f / system_fps))
        cout << "  Status: REAL-TIME CAPABLE (mean processing time < 1/dataset_fps)" << endl;
    else
        cout << "  Status: NOT REAL-TIME (mean processing time > 1/dataset_fps)" << endl;
    
    if(p99_track < (1.0f / system_fps))
        cout << "  P99 Latency: MEETS DEADLINE" << endl;
    else
        cout << "  P99 Latency: EXCEEDS DEADLINE" << endl;

    // Memory usage
#ifdef __linux__
    cout << endl;
    cout << "--- Memory Usage ---" << endl;
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    double ram_mb = usage.ru_maxrss / 1024.0;
    cout << "  Peak RAM: " << ram_mb << " MB" << endl;
#endif

    cout << endl;
    cout << "====================================================================" << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    
    // Save 3D point cloud map
    SLAM.SavePCL("MapPoints.pcd");
    cout << "Map saved to MapPoints.pcd" << endl;

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}

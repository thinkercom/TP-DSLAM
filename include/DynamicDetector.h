#ifndef DYNAMIC_DETECTOR_H
#define DYNAMIC_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

struct Detection
{
    int class_id;
    float conf;
    cv::Rect box;
    float dyn_prior;
};

class DynamicDetector
{
public:
    DynamicDetector(const std::string &model_path,
                    float conf_thres = 0.4f,
                    float score_thres = 0.25f,
                    float nms_thres = 0.45f,
                    int input_size = 640);
    ~DynamicDetector();

    // output dynamic prior CV_32FC1, range [0.1, 0.95]
    bool inferDynamicPrior(const cv::Mat &image, cv::Mat &dynamic_prior_map);

    bool inferDynamicMask(const cv::Mat &image, cv::Mat &dynamic_mask);

    // save prior map
    bool savePriorMapVisualization(const cv::Mat &prior_map,
                                   const std::string &save_path);

    // get prior map statistics
    void getPriorMapStats(const cv::Mat &prior_map,
                          float &mean_prior,
                          float &max_prior,
                          int &dynamic_pixel_count);

    const std::vector<Detection> &getDetections() const { return detections_; }

private:
    bool isDynamicClass(int class_id) const;
    float getDynamicPriorByClass(int class_id) const;

private:
    // ONNX Runtime
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<const char *> input_names_;
    std::vector<const char *> output_names_;
    std::string input_name_str_;
    std::vector<std::string> output_name_strs_;

    // config
    float conf_thres_;
    float score_thres_;
    float nms_thres_;
    int input_size_;

    // dectetion results
    std::vector<Detection> detections_;

    // mapping from class id to dynamic prior
    std::unordered_map<int, float> class_dyn_prior_;
};

#endif
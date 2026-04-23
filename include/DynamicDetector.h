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

    // 输出动态先验图：CV_32FC1, range [0.1, 0.95], 尺寸与输入图像一致
    bool inferDynamicPrior(const cv::Mat &image, cv::Mat &dynamic_prior_map);

    // 输出二值动态候选 mask: CV_8UC1, values {0, 255}
    bool inferDynamicMask(const cv::Mat &image, cv::Mat &dynamic_mask);

    // 【新增】保存概率热图可视化
    bool savePriorMapVisualization(const cv::Mat &prior_map,
                                   const std::string &save_path);

    // 【新增】获取先验图统计信息
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

    // 参数
    float conf_thres_;
    float score_thres_;
    float nms_thres_;
    int input_size_;

    // 检测结果
    std::vector<Detection> detections_;

    // 类别到动态先验分数的映射
    std::unordered_map<int, float> class_dyn_prior_;
};

#endif
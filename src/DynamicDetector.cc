#include "DynamicDetector.h"
#include <iostream>
#include <algorithm>
#include <numeric>

// ==================== 辅助函数 ====================

struct LetterBoxInfo
{
    float scale;
    int pad_w;
    int pad_h;
};

static cv::Mat letterbox(const cv::Mat &image, int new_shape, LetterBoxInfo &info)
{
    int width = image.cols;
    int height = image.rows;

    float r = std::min((float)new_shape / width, (float)new_shape / height);
    int new_unpad_w = int(std::round(width * r));
    int new_unpad_h = int(std::round(height * r));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_unpad_w, new_unpad_h));

    int dw = new_shape - new_unpad_w;
    int dh = new_shape - new_unpad_h;

    dw /= 2;
    dh /= 2;

    cv::Mat out;
    cv::copyMakeBorder(resized, out, dh, new_shape - new_unpad_h - dh,
                       dw, new_shape - new_unpad_w - dw,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    info.scale = r;
    info.pad_w = dw;
    info.pad_h = dh;
    return out;
}

static std::vector<float> blobFromImage(const cv::Mat &image)
{
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor_values(1 * 3 * rgb.rows * rgb.cols);
    std::vector<cv::Mat> chw(3);

    for (int i = 0; i < 3; ++i)
    {
        chw[i] = cv::Mat(rgb.rows, rgb.cols, CV_32F,
                         input_tensor_values.data() + i * rgb.rows * rgb.cols);
    }

    cv::split(rgb, chw);
    return input_tensor_values;
}

static cv::Rect scaleBoxToOriginal(float cx, float cy, float w, float h,
                                   const LetterBoxInfo &lb, int orig_w, int orig_h)
{
    float x1 = (cx - 0.5f * w - lb.pad_w) / lb.scale;
    float y1 = (cy - 0.5f * h - lb.pad_h) / lb.scale;
    float x2 = (cx + 0.5f * w - lb.pad_w) / lb.scale;
    float y2 = (cy + 0.5f * h - lb.pad_h) / lb.scale;

    x1 = std::max(0.f, std::min(x1, (float)(orig_w - 1)));
    y1 = std::max(0.f, std::min(y1, (float)(orig_h - 1)));
    x2 = std::max(0.f, std::min(x2, (float)(orig_w - 1)));
    y2 = std::max(0.f, std::min(y2, (float)(orig_h - 1)));

    return cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
}

static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

static cv::Mat decodeMask(const std::vector<float> &coeff,
                          const float *proto_data,
                          int mask_dim, int proto_h, int proto_w,
                          const cv::Rect &box,
                          const LetterBoxInfo &lb,
                          int orig_w, int orig_h,
                          int input_size = 640)
{
    // 1) coeff × proto -> [proto_h, proto_w]
    cv::Mat mask = cv::Mat::zeros(proto_h, proto_w, CV_32F);

    for (int c = 0; c < mask_dim; ++c)
    {
        cv::Mat proto(proto_h, proto_w, CV_32F, (void *)(proto_data + c * proto_h * proto_w));
        mask += coeff[c] * proto;
    }

    // sigmoid
    for (int y = 0; y < mask.rows; ++y)
    {
        float *p = mask.ptr<float>(y);
        for (int x = 0; x < mask.cols; ++x)
        {
            p[x] = sigmoid(p[x]);
        }
    }

    // 2) resize 到输入尺寸
    cv::Mat mask_up;
    cv::resize(mask, mask_up, cv::Size(input_size, input_size), 0, 0, cv::INTER_LINEAR);

    // 3) 去掉 letterbox padding
    int x0 = lb.pad_w;
    int y0 = lb.pad_h;
    int w = int(orig_w * lb.scale);
    int h = int(orig_h * lb.scale);

    cv::Rect roi(x0, y0, w, h);
    roi &= cv::Rect(0, 0, input_size, input_size);

    cv::Mat mask_crop = mask_up(roi).clone();

    // 4) resize 回原图
    cv::Mat mask_orig;
    cv::resize(mask_crop, mask_orig, cv::Size(orig_w, orig_h), 0, 0, cv::INTER_LINEAR);

    // 5) 二值化
    cv::Mat mask_bin;
    cv::threshold(mask_orig, mask_bin, 0.5, 255, cv::THRESH_BINARY);
    mask_bin.convertTo(mask_bin, CV_8U);

    // 6) 只保留 box 区域
    cv::Mat final_mask = cv::Mat::zeros(orig_h, orig_w, CV_8U);
    if (box.x >= 0 && box.y >= 0 && box.x + box.width <= orig_w && box.y + box.height <= orig_h)
    {
        mask_bin(box).copyTo(final_mask(box));
    }

    return final_mask;
}

// ==================== DynamicDetector 实现 ====================

DynamicDetector::DynamicDetector(const std::string &model_path,
                                 float conf_thres,
                                 float score_thres,
                                 float nms_thres,
                                 int input_size)
    : conf_thres_(conf_thres),
      score_thres_(score_thres),
      nms_thres_(nms_thres),
      input_size_(input_size)
{
    // 初始化 ONNX Runtime
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DYNAMIC_DETECTOR");

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), session_options);

    if (session_ == nullptr)
    {
        throw std::runtime_error("Failed to load ONNX model: " + model_path);
    }

    // 获取输入输出名称
    Ort::AllocatorWithDefaultOptions allocator;

    // 1. 处理输入名称
    auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_name_str_ = std::string(input_name_ptr.get());
    input_names_.push_back(input_name_str_.c_str());

    // 2. 处理输出名称
    size_t num_outputs = session_->GetOutputCount();
    output_name_strs_.clear();
    output_names_.clear();

    for (size_t i = 0; i < num_outputs; ++i)
    {
        auto out_name_ptr = session_->GetOutputNameAllocated(i, allocator);
        output_name_strs_.push_back(std::string(out_name_ptr.get()));
    }

    // 3. 将 std::string 的 c_str() 填入 const char* 向量供 ONNX 使用
    for (const auto &s : output_name_strs_)
    {
        output_names_.push_back(s.c_str());
    }

    // COCO 常见潜在动态类别 -> 动态先验
    class_dyn_prior_ = {
        {0, 0.95f},  // person
        {1, 0.75f},  // bicycle
        {2, 0.70f},  // car
        {3, 0.80f},  // motorcycle
        {5, 0.70f},  // bus
        {7, 0.70f},  // truck
        {15, 0.90f}, // cat
        {16, 0.90f}, // dog
        {17, 0.85f}, // horse
        {18, 0.85f}, // sheep
        {19, 0.85f}, // cow
        {21, 0.80f}, // bear
        {22, 0.80f}, // zebra
        {23, 0.80f}  // giraffe
    };
}

DynamicDetector::~DynamicDetector() = default;

bool DynamicDetector::isDynamicClass(int class_id) const
{
    return class_dyn_prior_.find(class_id) != class_dyn_prior_.end();
}

float DynamicDetector::getDynamicPriorByClass(int class_id) const
{
    auto it = class_dyn_prior_.find(class_id);
    if (it != class_dyn_prior_.end())
        return it->second;
    return 0.1f; // 默认静态先验
}

bool DynamicDetector::inferDynamicPrior(const cv::Mat &image, cv::Mat &dynamic_prior_map)
{
    detections_.clear();

    if (image.empty())
    {
        std::cerr << "Input image is empty." << std::endl;
        return false;
    }

    int orig_w = image.cols;
    int orig_h = image.rows;

    // ==================== 初始化动态先验图 ====================
    // 【关键】确保输出尺寸与输入图像完全一致
    dynamic_prior_map = cv::Mat(orig_h, orig_w, CV_32FC1, cv::Scalar(0.1f));

    // ==================== 预处理 ====================
    LetterBoxInfo lb;
    cv::Mat input_image = letterbox(image, input_size_, lb);
    std::vector<float> input_tensor_values = blobFromImage(input_image);

    // ==================== 创建输入 Tensor ====================
    std::vector<int64_t> input_shape = {1, 3, input_size_, input_size_};
    size_t input_tensor_size = input_tensor_values.size();

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size,
        input_shape.data(), input_shape.size());

    // ==================== 推理 ====================
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_.data(), &input_tensor, 1,
        output_names_.data(), output_names_.size());

    if (output_tensors.size() < 2)
    {
        std::cerr << "Expected 2 outputs (detections + proto), got "
                  << output_tensors.size() << std::endl;
        return false;
    }

    // ==================== 解析输出 ====================
    float *out0 = output_tensors[0].GetTensorMutableData<float>();
    float *proto = output_tensors[1].GetTensorMutableData<float>();

    auto shape0 = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto shape1 = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    int c = (int)shape0[1];
    int n = (int)shape0[2];

    int mask_dim = (int)shape1[1];
    int proto_h = (int)shape1[2];
    int proto_w = (int)shape1[3];

    // 计算类别数：C = 4 (bbox) + num_classes + mask_dim
    int num_classes = c - 4 - mask_dim;

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<std::vector<float>> coeffs_all;
    std::vector<float> dyn_priors;

    // ==================== 解析检测输出 ====================
    for (int i = 0; i < n; ++i)
    {
        const float *p = out0 + i; // 按 [C, N] 列优先访问

        float cx = p[0 * n];
        float cy = p[1 * n];
        float w = p[2 * n];
        float h = p[3 * n];

        // 找最大类别分数
        float best_score = 0.f;
        int best_class = -1;

        for (int cls = 0; cls < num_classes; ++cls)
        {
            float s = p[(4 + cls) * n];
            if (s > best_score)
            {
                best_score = s;
                best_class = cls;
            }
        }

        // 分数阈值过滤
        if (best_score < score_thres_)
            continue;

        // 只保留动态类别
        if (!isDynamicClass(best_class))
            continue;

        // 坐标还原到原图
        cv::Rect box = scaleBoxToOriginal(cx, cy, w, h, lb, orig_w, orig_h);
        if (box.width <= 1 || box.height <= 1)
            continue;

        // 提取 mask 系数
        std::vector<float> coeff(mask_dim);
        for (int k = 0; k < mask_dim; ++k)
        {
            coeff[k] = p[(4 + num_classes + k) * n];
        }

        boxes.push_back(box);
        scores.push_back(best_score);
        class_ids.push_back(best_class);
        coeffs_all.push_back(coeff);
        dyn_priors.push_back(getDynamicPriorByClass(best_class));
    }

    // ==================== NMS ====================
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, conf_thres_, nms_thres_, indices);

    // ==================== 生成动态先验概率热图 ====================
    // 【关键】使用实例 mask 精确更新每个像素的动态概率
    for (int idx : indices)
    {
        Detection det;
        det.class_id = class_ids[idx];
        det.conf = scores[idx];
        det.box = boxes[idx];
        det.dyn_prior = dyn_priors[idx];

        // 解码实例 mask（得到原图尺寸的二值 mask）
        cv::Mat instance_mask = decodeMask(
            coeffs_all[idx],
            proto,
            mask_dim, proto_h, proto_w,
            det.box,
            lb,
            orig_w, orig_h,
            input_size_);

        // 【关键】使用 mask 区域更新动态先验图
        // 对于重叠区域，取最大动态概率（最保守策略）
        for (int y = 0; y < orig_h; ++y)
        {
            for (int x = 0; x < orig_w; ++x)
            {
                if (instance_mask.at<uchar>(y, x) > 0)
                {
                    float &current_prior = dynamic_prior_map.at<float>(y, x);
                    current_prior = std::max(current_prior, det.dyn_prior);
                }
            }
        }

        detections_.push_back(det);
    }

    // ==================== 可选：高斯模糊使边界更平滑 ====================
    // 【关键】让动态/静态边界过渡更自然，避免硬切割
    if (!detections_.empty())
    {
        cv::GaussianBlur(dynamic_prior_map, dynamic_prior_map, cv::Size(5, 5), 1.5);

        // 确保值范围在 [0.1, 0.95] 之间
        cv::threshold(dynamic_prior_map, dynamic_prior_map, 0.95, 0.95, cv::THRESH_TRUNC);
        cv::threshold(dynamic_prior_map, dynamic_prior_map, 0.1, 0.1, cv::THRESH_TOZERO);
    }

    return true;
}

bool DynamicDetector::inferDynamicMask(const cv::Mat &image, cv::Mat &dynamic_mask)
{
    cv::Mat prior_map;
    if (!inferDynamicPrior(image, prior_map))
        return false;

    // 二值化：先验分数 > 0.5 视为动态区域
    dynamic_mask = cv::Mat::zeros(prior_map.size(), CV_8UC1);
    for (int y = 0; y < prior_map.rows; ++y)
    {
        for (int x = 0; x < prior_map.cols; ++x)
        {
            if (prior_map.at<float>(y, x) > 0.5f)
                dynamic_mask.at<uchar>(y, x) = 255;
        }
    }
    return true;
}

// ==================== 新增：可视化保存概率热图 ====================
bool DynamicDetector::savePriorMapVisualization(const cv::Mat &prior_map,
                                                const std::string &save_path)
{
    if (prior_map.empty() || prior_map.type() != CV_32FC1)
    {
        std::cerr << "Invalid prior map for visualization." << std::endl;
        return false;
    }

    // 1. 归一化到 [0, 255]
    cv::Mat prior_u8;
    prior_map.convertTo(prior_u8, CV_8UC1, 255.0);

    // 2. 应用伪彩色映射
    cv::Mat color_map;
    cv::applyColorMap(prior_u8, color_map, cv::COLORMAP_JET);

    // 3. 保存
    cv::imwrite(save_path, color_map);

    std::cout << "Saved prior map visualization: " << save_path << std::endl;
    return true;
}

// ==================== 新增：获取先验图统计信息 ====================
void DynamicDetector::getPriorMapStats(const cv::Mat &prior_map,
                                       float &mean_prior,
                                       float &max_prior,
                                       int &dynamic_pixel_count)
{
    if (prior_map.empty() || prior_map.type() != CV_32FC1)
    {
        mean_prior = 0.1f;
        max_prior = 0.1f;
        dynamic_pixel_count = 0;
        return;
    }

    // 计算均值和标准差
    cv::Scalar mean_val, std_val;
    cv::meanStdDev(prior_map, mean_val, std_val);
    mean_prior = static_cast<float>(mean_val[0]);

    // 【修复】使用 double 临时变量接收 minMaxLoc 的结果
    double min_val = 0.0;
    double max_val = 0.0;
    cv::minMaxLoc(prior_map, &min_val, &max_val);
    max_prior = static_cast<float>(max_val);

    // 统计动态像素数量
    dynamic_pixel_count = cv::countNonZero(prior_map > 0.5f);
}
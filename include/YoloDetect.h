
#ifndef YOLO_DETECT_H
#define YOLO_DETECT_H

#include <opencv2/opencv.hpp>
#include <torch/script.h>
#include <algorithm>
#include <iostream>
#include <utility>
#include <time.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
using namespace std;

class YoloDetection
{
public:
    YoloDetection();
    ~YoloDetection();
    void GetImage(cv::Mat& RGB);
    void ClearImage();
    bool Detect();
    void ClearArea();

    float generate_scale(cv::Mat& image, const std::vector<int>& target_size);
    float letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size);
    torch::Tensor xyxy2xywh(const torch::Tensor& x);
    torch::Tensor xywh2xyxy(const torch::Tensor& x);
    torch::Tensor nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold);
    torch::Tensor non_max_suppression(torch::Tensor& prediction, float conf_thres = 0.4, float iou_thres = 0.5, int max_det = 300) ;
    torch::Tensor scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape);
    
    vector<cv::Rect2i> mvPersonArea = {};
    //vector<torch::Tensor> non_max_suppression(torch::Tensor preds, float score_thresh=0.5, float iou_thresh=0.5);

public:
    cv::Mat mRGB;
    torch::jit::script::Module mModule;
    std::vector<std::string> mClassnames;

    // 6-28
    vector<string> mvDynamicNames;
    vector<cv::Rect2i> mvDynamicArea;
    map<string, vector<cv::Rect2i>> mmDetectMap;


};


#endif //YOLO_DETECT_H

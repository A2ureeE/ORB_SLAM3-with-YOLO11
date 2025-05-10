#include <YoloDetect.h>
// 使用torch::indexing命名空间中的Slice和None
using torch::indexing::Slice;
using torch::indexing::None;

YoloDetection::YoloDetection()//
{
    torch::jit::setTensorExprFuserEnabled(false);//关闭tensor表达式融合器
    //torch::jit::script::Module mModule;
    mModule = torch::jit::load("/home/zhang/modified_YOLOORB_SLAM3_improved/yolo/yolo11n.torchscript");       //模型加载
    mModule.eval();//模型设置为评估模式
    mModule.to(torch::kFloat);//模型转换为float32
    std::ifstream f("coco.names");//加载类别名称
    std::string name = "";//空类别名称
    while (std::getline(f, name))
    {
        mClassnames.push_back(name);//类别名称存入mClassnames
    }
    mvDynamicNames = {"person", "car", "motorbike", "bus", "train", "truck", "boat", "bird", "cat",
                      "dog", "horse", "sheep", "crow", "bear","tvmonitor"};//筛选动态物体类别
}

YoloDetection::~YoloDetection()
{

}

bool YoloDetection::Detect()//检测器
{
    torch::Device device(torch::kCPU);
    cv::Mat img;
   

    if(mRGB.empty())
    {
        std::cout << "Read RGB failed!" << std::endl;
        return false;
    }
    //检测输入图像的通道数
    if (mRGB.channels() == 1)
    {
        cv::cvtColor(mRGB, mRGB, cv::COLOR_GRAY2RGB);
    }


    cv::resize(mRGB, img, cv::Size(640, 640));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    // letterbox(mRGB, img, {640, 640}); // 调整图像大小并添加填充
    torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte).clone();//图像数据转换为张量

    //imgTensor = imgTensor.toType(torch::kFloat).div(255); // 将张量归一化到[0,1]范围
    torch::Tensor keep= imgTensor.toType(torch::kFloat); // 转换张量数据类型为浮点型
    keep = keep.permute({2,0,1});
    keep = keep.unsqueeze(0); 

    keep = keep.div(255); 

    
    
    

    std::vector<torch::jit::IValue> inputs {keep}; // 创建输入向量



    // // Preparing input tensor
    // cv::resize(mRGB, img, cv::Size(640, 380));//图像尺寸调整
    // std::cout<<"img ready"<<std::endl;
    // //v::Mat bgr;//图像颜色通道调整
    // //letter_box(mRGB, img, {640, 640});
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);//图像颜色通道调整
    // torch::Tensor imgTensor = torch::from_blob(img.data, {img.rows, img.cols,3},torch::kByte);//图像数据转换为张量
    // imgTensor = imgTensor.permute({2,0,1}); // 调整张量维度顺序(调整维度顺序为 (C, H, W)（PyTorch 的 NCHW 格式）。)
    // //如果成功了输出success
    // imgTensor = imgTensor.toType(torch::kFloat); // 转换张量数据类型为浮点型
    // imgTensor = imgTensor.div(255); // 将张量归一化到[0,1]范围
    // imgTensor = imgTensor.unsqueeze(0); // 在第0维增加一个维度(形状变为 (1, C, H, W)。)
    // std::cout<<"all right"<<std::endl;
 
    
 
// 针对新版YOLO模块的改动
    torch::Tensor preds;
    preds = mModule.forward(inputs).toTensor().cpu();//模型前向推理
//NMS
    auto dets = non_max_suppression(preds)[0];

    auto boxes = dets.index({Slice(), Slice(None, 4)});//获取边界框

    //输出boxes内容

    //dets.index_put_({Slice(), Slice(None, 4)}, scale_boxes({img.rows, img.cols}, boxes, {mRGB.rows, mRGB.cols}));
    if (dets.size(0)>0)
    {
        //遍历检测结果
        for(auto i=0;i<dets.size(0);i++)
        {   
            int left = dets[i][0].item().toFloat()* mRGB.cols / 640;
            int top = dets[i][1].item().toFloat()* mRGB.rows / 640;
            int right = dets[i][2].item().toFloat()* mRGB.cols / 640;
            int bottom = dets[i][3].item().toFloat()* mRGB.rows/ 640;
            int classID = dets[i][5].item().toInt(); // 获取类别ID
            cv::Rect2i DetectArea(left, top, (right - left), (bottom - top)); // 创建检测区域

            cout<<"DetectArea: left="<<left<<", top="<<top<<", (right - left)="<<(right - left)<<", (bottom - top)="<<(bottom - top)<<", classID="<<classID<<endl; // 输出检测区域信息
        cout<<"mClassnames[classID]] = "<<mClassnames[classID]<<endl; // 输出类别名称
        mmDetectMap[mClassnames[classID]].push_back(DetectArea); // 将检测区域存入映射表
            if (count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[classID])) // 判断是否为动态物体
        {
                cv::Rect2i DynamicArea(left, top, (right - left), (bottom - top)); // 创建动态物体区域
            mvDynamicArea.push_back(DynamicArea); // 将动态物体区域存入动态区域列表
        }
    }
    if (mvDynamicArea.size() == 0) // 如果没有检测到动态物体
        {
            cv::Rect2i tDynamicArea(1, 1, 1, 1); // 创建一个默认的动态物体区域
            mvDynamicArea.push_back(tDynamicArea); // 将默认的动态物体区域存入动态区域列表
        }
    }
    return true;
        
    
    //auto dets = preds.squeeze(0);


    
// // //老YOLO模块
//     torch::Tensor preds = mModule.forward({imgTensor}).toTuple()->elements()[0].toTensor();
//     //preds的维度
//     std::cout << "preds的维度：" << preds.sizes() << std::endl;
//     std::vector<torch::Tensor> dets = YoloDetection::non_max_suppression(preds, 0.4, 0.5);
//     std::cout << "dets的维度：" << dets.size() << std::endl;
//     if (dets.size() > 0)
//     {
//         // Visualize result
//         for (size_t i=0; i < dets[0].sizes()[0]; ++ i) // 遍历检测结果
//         {
//             float left = dets[0][i][0].item().toFloat() * mRGB.cols / 640; // 计算左边界
//             float top = dets[0][i][1].item().toFloat() * mRGB.rows / 384; // 计算上边界
//             float right = dets[0][i][2].item().toFloat() * mRGB.cols / 640; // 计算右边界
//             float bottom = dets[0][i][3].item().toFloat() * mRGB.rows / 384; // 计算下边界
//             int classID = dets[0][i][5].item().toInt(); // 获取类别ID

//             cv::Rect2i DetectArea(left, top, (right - left), (bottom - top)); // 创建检测区域
//             cout<<"DetectArea: left="<<left<<", top="<<top<<", (right - left)="<<(right - left)<<", (bottom - top)="<<(bottom - top)<<", classID="<<classID<<endl; // 输出检测区域信息
//         cout<<"mClassnames[classID]] = "<<mClassnames[classID]<<endl; // 输出类别名称
//         mmDetectMap[mClassnames[classID]].push_back(DetectArea); // 将检测区域存入映射表

//         if (std::count(mvDynamicNames.begin(), mvDynamicNames.end(), mClassnames[classID])) // 判断是否为动态物体
//         {
//                 cv::Rect2i DynamicArea(left, top, (right - left), (bottom - top)); // 创建动态物体区域
//             mvDynamicArea.push_back(DynamicArea); // 将动态物体区域存入动态区域列表
//         }

//     }
//     if (mvDynamicArea.size() == 0) // 如果没有检测到动态物体
//         {
//             cv::Rect2i tDynamicArea(1, 1, 1, 1); // 创建一个默认的动态物体区域
//             mvDynamicArea.push_back(tDynamicArea); // 将默认的动态物体区域存入动态区域列表
//         }
//     }
    return true;
}



//************************************************************************************************************************************************************************* */

float YoloDetection::generate_scale(cv::Mat& image, const std::vector<int>& target_size) {
    int origin_w = image.cols; // 原始图像宽度
    int origin_h = image.rows; // 原始图像高度

    int target_h = target_size[0]; // 目标高度
    int target_w = target_size[1]; // 目标宽度

    float ratio_h = static_cast<float>(target_h) / static_cast<float>(origin_h); // 高度缩放比例
    float ratio_w = static_cast<float>(target_w) / static_cast<float>(origin_w); // 宽度缩放比例
    float resize_scale = std::min(ratio_h, ratio_w); // 选择较小的缩放比例
    return resize_scale; // 返回缩放比例
}

float YoloDetection::letterbox(cv::Mat &input_image, cv::Mat &output_image, const std::vector<int> &target_size) {
    if (input_image.cols == target_size[1] && input_image.rows == target_size[0]) {
        if (input_image.data == output_image.data) {
            return 1.;
        } else {
            output_image = input_image.clone();
            return 1.;
        }
    }

    float resize_scale = generate_scale(input_image, target_size);
    int new_shape_w = std::round(input_image.cols * resize_scale);
    int new_shape_h = std::round(input_image.rows * resize_scale);
    float padw = (target_size[1] - new_shape_w) / 2.;
    float padh = (target_size[0] - new_shape_h) / 2.;

    int top = std::round(padh - 0.1);
    int bottom = std::round(padh + 0.1);
    int left = std::round(padw - 0.1);
    int right = std::round(padw + 0.1);

    cv::resize(input_image, output_image,
               cv::Size(new_shape_w, new_shape_h),
               0, 0, cv::INTER_AREA);

    cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114.));
    return resize_scale;
}

torch::Tensor YoloDetection::xyxy2xywh(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    y.index_put_({"...", 0}, (x.index({"...", 0}) + x.index({"...", 2})).div(2));
    y.index_put_({"...", 1}, (x.index({"...", 1}) + x.index({"...", 3})).div(2));
    y.index_put_({"...", 2}, x.index({"...", 2}) - x.index({"...", 0}));
    y.index_put_({"...", 3}, x.index({"...", 3}) - x.index({"...", 1}));
    return y;
}


torch::Tensor YoloDetection::xywh2xyxy(const torch::Tensor& x) {
    auto y = torch::empty_like(x);
    auto dw = x.index({"...", 2}).div(2);
    auto dh = x.index({"...", 3}).div(2);
    y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
    y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
    y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
    y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
    return y;
}



// Reference: https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
torch::Tensor YoloDetection::nms(const torch::Tensor& bboxes, const torch::Tensor& scores, float iou_threshold) {
    if (bboxes.numel() == 0)
        return torch::empty({0}, bboxes.options().dtype(torch::kLong));

    auto x1_t = bboxes.select(1, 0).contiguous();
    auto y1_t = bboxes.select(1, 1).contiguous();
    auto x2_t = bboxes.select(1, 2).contiguous();
    auto y2_t = bboxes.select(1, 3).contiguous();

    torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

    auto order_t = std::get<1>(
        scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

    auto ndets = bboxes.size(0);
    torch::Tensor suppressed_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kByte));
    torch::Tensor keep_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kLong));

    auto suppressed = suppressed_t.data_ptr<uint8_t>();
    auto keep = keep_t.data_ptr<int64_t>();
    auto order = order_t.data_ptr<int64_t>();
    auto x1 = x1_t.data_ptr<float>();
    auto y1 = y1_t.data_ptr<float>();
    auto x2 = x2_t.data_ptr<float>();
    auto y2 = y2_t.data_ptr<float>();
    auto areas = areas_t.data_ptr<float>();

    int64_t num_to_keep = 0;

    for (int64_t _i = 0; _i < ndets; _i++) {
        auto i = order[_i];
        if (suppressed[i] == 1)
            continue;
        keep[num_to_keep++] = i;
        auto ix1 = x1[i];
        auto iy1 = y1[i];
        auto ix2 = x2[i];
        auto iy2 = y2[i];
        auto iarea = areas[i];

        for (int64_t _j = _i + 1; _j < ndets; _j++) {
        auto j = order[_j];
        if (suppressed[j] == 1)
            continue;
        auto xx1 = std::max(ix1, x1[j]);
        auto yy1 = std::max(iy1, y1[j]);
        auto xx2 = std::min(ix2, x2[j]);
        auto yy2 = std::min(iy2, y2[j]);

        auto w = std::max(static_cast<float>(0), xx2 - xx1);
        auto h = std::max(static_cast<float>(0), yy2 - yy1);
        auto inter = w * h;
        auto ovr = inter / (iarea + areas[j] - inter);
        if (ovr > iou_threshold)
            suppressed[j] = 1;
        }
    }
    return keep_t.narrow(0, 0, num_to_keep);
}

// 非极大值抑制
torch::Tensor YoloDetection::non_max_suppression(torch::Tensor& prediction, float conf_thres, float iou_thres , int max_det) {
    auto bs = prediction.size(0); // 批次大小
    auto nc = prediction.size(1) - 4; // 类别数量
    auto nm = prediction.size(1) - nc - 4; // 掩码数量
    auto mi = 4 + nc; // 类别索引
    auto xc = prediction.index({Slice(), Slice(4, mi)}).amax(1) > conf_thres; // 置信度阈值过滤

    prediction = prediction.transpose(-1, -2); // 转置张量
    prediction.index_put_({"...", Slice({None, 4})}, xywh2xyxy(prediction.index({"...", Slice(None, 4)}))); // 转换边界框格式

    std::vector<torch::Tensor> output; // 输出结果
    for (int i = 0; i < bs; i++) {
        output.push_back(torch::zeros({0, 6 + nm}, prediction.device())); // 初始化输出张量
    }

    for (int xi = 0; xi < prediction.size(0); xi++) {
        auto x = prediction[xi]; // 获取当前批次的预测结果
        x = x.index({xc[xi]}); // 过滤置信度低于阈值的预测结果
        auto x_split = x.split({4, nc, nm}, 1); // 分割张量
        auto box = x_split[0], cls = x_split[1], mask = x_split[2]; // 获取边界框、类别和掩码
        auto [conf, j] = cls.max(1, true); // 获取最大置信度和对应的类别索引
        x = torch::cat({box, conf, j.toType(torch::kFloat), mask}, 1); // 拼接张量
        x = x.index({conf.view(-1) > conf_thres}); // 过滤置信度低于阈值的预测结果
        int n = x.size(0); // 获取剩余预测结果数量
        if (!n) { continue; } // 如果没有剩余预测结果，则跳过

        // 非极大值抑制
        auto c = x.index({Slice(), Slice{5, 6}}) * 7680; // 类别偏移
        auto boxes = x.index({Slice(), Slice(None, 4)}) + c; // 获取边界框
        auto scores = x.index({Slice(), 4}); // 获取置信度
        auto i = nms(boxes, scores, iou_thres); // 执行非极大值抑制
        i = i.index({Slice(None, max_det)}); // 限制最大检测数量
        output[xi] = x.index({i}); // 保存结果
    }

    return torch::stack(output); // 返回结果
}

// 缩放边界框
torch::Tensor YoloDetection::scale_boxes(const std::vector<int>& img1_shape, torch::Tensor& boxes, const std::vector<int>& img0_shape) {
    // 计算缩放比例
    auto gain = (std::min)((float)img1_shape[0] / img0_shape[0], (float)img1_shape[1] / img0_shape[1]);
    // 计算填充
    auto pad0 = std::round((float)(img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
    auto pad1 = std::round((float)(img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

    // 调整边界框
    boxes.index_put_({"...", 0}, boxes.index({"...", 0}) - pad0);
    boxes.index_put_({"...", 2}, boxes.index({"...", 2}) - pad0);
    boxes.index_put_({"...", 1}, boxes.index({"...", 1}) - pad1);
    boxes.index_put_({"...", 3}, boxes.index({"...", 3}) - pad1);
    boxes.index_put_({"...", Slice(None, 4)}, boxes.index({"...", Slice(None, 4)}).div(gain));
    return boxes;
}


























// //FOR V5
// vector<torch::Tensor> YoloDetection::non_max_suppression(torch::Tensor preds, float score_thresh, float iou_thresh)
// {
//     std::vector<torch::Tensor> output;
//     for (size_t i=0; i < preds.sizes()[0]; ++i)
//     {
//         torch::Tensor pred = preds.select(0, i);//选择第i个检测结果

//         // Filter by scores
//         torch::Tensor scores = pred.select(1, 4) * std::get<0>( torch::max(pred.slice(1, 5, pred.sizes()[1]), 1));//
//         pred = torch::index_select(pred, 0, torch::nonzero(scores > score_thresh).select(1, 0));
//         if (pred.sizes()[0] == 0) continue;

//         // (center_x, center_y, w, h) to (left, top, right, bottom)
//         pred.select(1, 0) = pred.select(1, 0) - pred.select(1, 2) / 2;
//         pred.select(1, 1) = pred.select(1, 1) - pred.select(1, 3) / 2;
//         pred.select(1, 2) = pred.select(1, 0) + pred.select(1, 2);
//         pred.select(1, 3) = pred.select(1, 1) + pred.select(1, 3);

//         // Computing scores and classes
//         std::tuple<torch::Tensor, torch::Tensor> max_tuple = torch::max(pred.slice(1, 5, pred.sizes()[1]), 1);
//         pred.select(1, 4) = pred.select(1, 4) * std::get<0>(max_tuple);
//         pred.select(1, 5) = std::get<1>(max_tuple);

//         torch::Tensor  dets = pred.slice(1, 0, 6);

//         torch::Tensor keep = torch::empty({dets.sizes()[0]});
//         torch::Tensor areas = (dets.select(1, 3) - dets.select(1, 1)) * (dets.select(1, 2) - dets.select(1, 0));
//         std::tuple<torch::Tensor, torch::Tensor> indexes_tuple = torch::sort(dets.select(1, 4), 0, 1);
//         torch::Tensor v = std::get<0>(indexes_tuple);
//         torch::Tensor indexes = std::get<1>(indexes_tuple);
//         int count = 0;
//         while (indexes.sizes()[0] > 0)
//         {
//             keep[count] = (indexes[0].item().toInt());
//             count += 1;

//             // Computing overlaps
//             torch::Tensor lefts = torch::empty(indexes.sizes()[0] - 1);
//             torch::Tensor tops = torch::empty(indexes.sizes()[0] - 1);
//             torch::Tensor rights = torch::empty(indexes.sizes()[0] - 1);
//             torch::Tensor bottoms = torch::empty(indexes.sizes()[0] - 1);
//             torch::Tensor widths = torch::empty(indexes.sizes()[0] - 1);
//             torch::Tensor heights = torch::empty(indexes.sizes()[0] - 1);
//             for (size_t i=0; i<indexes.sizes()[0] - 1; ++i)
//             {
//                 lefts[i] = std::max(dets[indexes[0]][0].item().toFloat(), dets[indexes[i + 1]][0].item().toFloat());
//                 tops[i] = std::max(dets[indexes[0]][1].item().toFloat(), dets[indexes[i + 1]][1].item().toFloat());
//                 rights[i] = std::min(dets[indexes[0]][2].item().toFloat(), dets[indexes[i + 1]][2].item().toFloat());
//                 bottoms[i] = std::min(dets[indexes[0]][3].item().toFloat(), dets[indexes[i + 1]][3].item().toFloat());
//                 widths[i] = std::max(float(0), rights[i].item().toFloat() - lefts[i].item().toFloat());
//                 heights[i] = std::max(float(0), bottoms[i].item().toFloat() - tops[i].item().toFloat());
//             }
//             torch::Tensor overlaps = widths * heights;


//             torch::Tensor ious = overlaps / (areas.select(0, indexes[0].item().toInt()) + torch::index_select(areas, 0, indexes.slice(0, 1, indexes.sizes()[0])) - overlaps);
//             indexes = torch::index_select(indexes, 0, torch::nonzero(ious <= iou_thresh).select(1, 0) + 1);
//         }
//         keep = keep.toType(torch::kInt64);
//         output.push_back(torch::index_select(dets, 0, keep.slice(0, 0, count)));
//     }
//     return output;
// }

void YoloDetection::GetImage(cv::Mat &RGB)
{
    mRGB = RGB;
}

void YoloDetection::ClearImage()
{
    mRGB = 0;
}

void YoloDetection::ClearArea()
{
    mvPersonArea.clear();
}




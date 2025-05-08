import cv2
import torch
from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11s.pt")

# Export the model to TorchScript format
#nms=True表示在导出时启用非极大值抑制,并且参数设置为iou=0.5
model.export(format="torchscript",imgsz=[640,640])  # creates 'yolo11s.torchscript'

# Load the exported TorchScript model
torchscript_model = YOLO("yolo11s.torchscript")
# Run inference
print("success")
testmodel="./bus.jpg"



print("success")

results = torchscript_model(testmodel)
print("success")

for result in results:
    boxes = result.boxes.xyxy  # Get the bounding boxes
    print("检测框的坐标：")
    for box in boxes:
        print(box)  # Print each bounding box

#检测输出的模型的信息和其输出每层的维度信息
# Print the results
#model = torch.jit.load("yolo11s.torchscript.pt")

# 生成测试输入
img = torch.randn(1, 3, 640, 640)

# 运行推理
with torch.no_grad():
    output = torchscript_model(img)
    output_torch = torch.tensor(output)  # 将列表转换为 PyTorch 张量
    print("看我看我",output_torch.shape)


# 检查输出类型
if isinstance(output, tuple):
    print(f"输出为元组，包含 {len(output)} 个张量")
    # Print the shape of each tensor in the tuple
    for i, tensor in enumerate(output):
        print(f"Tensor {i} 的形状: {tensor.shape}")
        # Print the shape of each tensor
        print(f"Tensor {i} 的形状: {tensor.shape}")
else :
        print("输出不是元组！需要调整模型结构")




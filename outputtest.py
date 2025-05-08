from tkinter.scrolledtext import example

import torch
from ultralytics import YOLO

model=torch.jit.load("yolo11s.torchscript")
example=torch.rand(1,3,640,384)
outputs=model(example)
print([output.shape for output in outputs])
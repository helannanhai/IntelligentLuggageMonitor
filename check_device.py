import torch
from ultralytics import YOLO

print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
try:
    m = YOLO('yolov8n-seg.pt')
    # ultralytics model has .model.device or .device
    dev = getattr(m.model, 'device', None)
    print('model_device', dev)
except Exception as e:
    print('failed to init YOLO model:', e)

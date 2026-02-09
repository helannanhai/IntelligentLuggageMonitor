from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model(
    source="Dataset/JiMeng/2.mp4",
    save=True,
    project=r"test_video/outputs",
    name="test2"
)
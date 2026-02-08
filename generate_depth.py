import cv2
import torch
import numpy as np


# === Input video path ===
video_path = "Hailuo_Video_Create a CCTV footage of peopl_398561169233838086.mp4"

# === 1. Read the first frame ===
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Failed to read the first frame. Please check the video path.")

print("First frame successfully loaded. Generating depth map...")

# Optional: save the first frame
# cv2.imwrite("first_frame_rgb.jpg", frame)

# === 2. Convert BGR to RGB ===
img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# === 3. Load MiDaS / DPT depth model ===
model_type = "DPT_Hybrid"
model = torch.hub.load("intel-isl/MiDaS", model_type)
model.eval()

# === 4. Apply input transform ===
transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
input_tensor = transform(img_rgb).unsqueeze(0)

# === 5. Predict depth ===
with torch.no_grad():
    depth = model(input_tensor)[0].cpu().numpy()

print("Depth map shape:", depth.shape)

# === 6. Save as .npy depth map (used by your main script) ===
np.save("first_frame_depth.npy", depth)

# Also save a normalized preview for visualization
depth_norm = (depth - depth.min()) / (depth.max() - depth.min())
depth_img = (depth_norm * 255).astype(np.uint8)
cv2.imwrite("first_frame_depth_preview.png", depth_img)

print("Generated: first_frame_depth.npy and first_frame_depth_preview.png")

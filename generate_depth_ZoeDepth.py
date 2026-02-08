import cv2
import torch
import numpy as np
from PIL import Image

from zoedepth.utils.misc import pil_to_batched_tensor, get_image_from_url, save_raw_16bit, colorize
# If your original code had imports from builder/config, keep them:
from zoedepth.models.builder import build_model
from zoedepth.utils.config import get_config


# ============================
# Load ZoeDepth model (unchanged)
# ============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config = get_config("zoedepth", "infer")
model_zoe_n = build_model(config)
state_dict = torch.hub.load_state_dict_from_url(config["checkpoint_uri"], map_location=DEVICE)
model_zoe_n.load_state_dict(state_dict)
model_zoe_n = model_zoe_n.to(DEVICE)
model_zoe_n.eval()


# ============================
# MINIMAL CHANGE:
# Read first frame from your video
# ============================
video_path = "Hailuo_Video_Create a CCTV footage of peopl_398561169233838086.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
cap.release()

if not success:
    raise RuntimeError("Failed to read first frame from the video.")

# Convert OpenCV BGR → PIL RGB
image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# ============================
# The rest is EXACTLY the same as your official sample
# ============================

# Numpy output
depth_numpy = model_zoe_n.infer_pil(image)

# 16-bit PIL depth image
depth_pil = model_zoe_n.infer_pil(image, output_type="pil")

# Tensor output
depth_tensor = model_zoe_n.infer_pil(image, output_type="tensor")

# Save raw 16-bit depth
save_raw_16bit(depth_pil, "depth_raw_16bit.png")

# Colorize and save
colored = colorize(depth_numpy)
Image.fromarray(colored).save("depth_colored.png")

# Save numpy depth (for your adaptive-radius script)
np.save("first_frame_depth.npy", depth_numpy)

print("Done: first_frame_depth.npy, depth_raw_16bit.png, depth_colored.png")

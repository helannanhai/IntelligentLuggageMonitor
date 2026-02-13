"""
配置文件 - 所有参数集中管理
"""

# ----------------------------
# 模型和视频配置
# ----------------------------
MODEL_PATH = "yolov8n-seg.pt"
VIDEO_PATH = r"Dataset\JiMeng\1_4.mov"
VIDEO_PATH1 = r"Dataset\JiMeng\1_3.mov"
VIDEO_PATH1 = r""
VIDEO_PATH1 = r""
VIDEO_PATH1 = r""
VIDEO_PATH1 = r""
VIDEO_PATH1 = r""
VIDEO_PATH1 = r""
VIDEO_PATH1 = r""
VIDEO_PATH1 = r""

# Output Settings
SAVE_OUTPUT_VIDEO = 1
OUTPUT_VIDEO_PATH = r"test_video\outputs\output_result.mp4"

OUTPUT_VIDEO_CODEC = "mp4v"

# ----------------------------
# Detection Configuration
# ----------------------------
PERSON_CLASS_ID = 0
LUGGAGE_CLASS_IDS = {24, 26, 28}  # backpack, handbag, suitcase

CONF_THRES = 0.35
IOU_THRES = 0.5
IMGSZ = 640

# ----------------------------
# Continuity Enhancement
# ----------------------------
MAX_LOST_FRAMES = 30  # Max wait time for stable objects (existed > 10 frames)
SHORT_LIVED_LOST_FRAMES = 3  # Max wait time for short-lived objects (existed < 10 frames)
STABLE_THRESHOLD_FRAMES = 8  # Frames threshold to be considered "stable"
POSITION_SMOOTHING = 0.3
MIN_CONSECUTIVE_FRAMES = 15  # Object must appear consecutively for N frames to be confirmed

# ----------------------------
# Display Settings
# ----------------------------
MASK_ALPHA = 0.35
LABEL_FONT_SIZE = 0.5
LABEL_FONT_THICKNESS = 1
DISTANCE_FONT_SIZE = 0.4
DISTANCE_FONT_THICKNESS = 1

# ----------------------------
# Ownership Detection
# ----------------------------
OWNER_DISTANCE_THRESHOLD = 350
OWNER_DISTANCE_SCALE_FACTOR = 2.0
OWNER_CONFIRM_SECONDS = 2.0
ABANDON_CONFIRM_SECONDS = 3.0
OWNER_HISTORY_FRAMES = 60
OWNER_CHANGE_THRESHOLD = 0.7

# Runtime-calculated frame thresholds
OWNER_CONFIRM_FRAMES = None
ABANDON_CONFIRM_FRAMES = None

# ----------------------------
# Person Re-identification
# ----------------------------
PERSON_FEATURE_HISTORY_FRAMES = 600
PERSON_OWNER_HISTORY_FRAMES = 1800
PERSON_REID_SIMILARITY_THRESHOLD = 0.70
PERSON_POSITION_REENTRY_THRESHOLD = 300

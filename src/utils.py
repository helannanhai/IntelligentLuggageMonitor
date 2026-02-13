"""
Utility Functions Module
"""
import os
import cv2
import numpy as np
from config import MASK_ALPHA


def parse_box(box, masks_data=None, idx=None):
    """Parse YOLO detection box, return (conf, cls_id, track_id, bbox, mask)"""
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
    track_id = int(box.id.item()) if box.id is not None else None
    mask_float = None
    if masks_data is not None and idx is not None:
        mask_float = masks_data[idx].detach().cpu().numpy()
    return conf, cls_id, track_id, (x1, y1, x2, y2), mask_float


def center_from_bbox(bbox):
    """Calculate bounding box center point"""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_unique_filename(filepath):
    """Generate unique filename to avoid overwriting existing files"""
    if not os.path.exists(filepath):
        return filepath
    
    directory = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    name, ext = os.path.splitext(filename)
    
    counter = 1
    while True:
        new_filename = f"{name}_{counter}{ext}"
        new_filepath = os.path.join(directory, new_filename) if directory else new_filename
        if not os.path.exists(new_filepath):
            return new_filepath
        counter += 1


def smooth_bbox(current_bbox, previous_bbox, alpha=0.3):
    """Apply exponential smoothing to bounding box coordinates to reduce jitter"""
    if previous_bbox is None:
        return current_bbox
    
    x1, y1, x2, y2 = current_bbox
    px1, py1, px2, py2 = previous_bbox
    
    sx1 = alpha * x1 + (1 - alpha) * px1
    sy1 = alpha * y1 + (1 - alpha) * py1
    sx2 = alpha * x2 + (1 - alpha) * px2
    sy2 = alpha * y2 + (1 - alpha) * py2
    
    return (sx1, sy1, sx2, sy2)


def overlay_mask(frame_bgr, mask_float, color_bgr, alpha=MASK_ALPHA):
    """Overlay segmentation mask on image"""
    h, w = frame_bgr.shape[:2]
    mask_u8 = (mask_float > 0.5).astype(np.uint8) * 255
    if mask_u8.shape[0] != h or mask_u8.shape[1] != w:
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = frame_bgr.copy()
    overlay[mask_u8 > 0] = color_bgr
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0)

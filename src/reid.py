"""
Person Re-identification Module - Re-ID Functionality
"""
import math
import cv2
import numpy as np
from config import (PERSON_FEATURE_HISTORY_FRAMES, PERSON_OWNER_HISTORY_FRAMES,
                   PERSON_REID_SIMILARITY_THRESHOLD, PERSON_POSITION_REENTRY_THRESHOLD)


def extract_person_features(frame, bbox):
    """Extract visual features (color histogram) from person bounding box"""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    
    # Calculate HSV color histogram
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv_roi], [0], None, [18], [0, 180])
    hist_s = cv2.calcHist([hsv_roi], [1], None, [10], [0, 256])
    hist_v = cv2.calcHist([hsv_roi], [2], None, [8], [0, 256])
    
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    hist_v = cv2.normalize(hist_v, hist_v).flatten()
    
    feature_vector = np.concatenate([hist_h, hist_s, hist_v])
    
    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    area = (x2 - x1) * (y2 - y1)
    height = y2 - y1
    
    return {
        "feature_vector": feature_vector,
        "center": center,
        "area": area,
        "height": height,
        "bbox": bbox
    }


def compute_feature_similarity(feat1, feat2):
    """Calculate cosine similarity between two feature vectors (0-1 range)"""
    if feat1 is None or feat2 is None:
        return 0.0
    
    v1 = feat1.get("feature_vector")
    v2 = feat2.get("feature_vector")
    
    if v1 is None or v2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    return max(0.0, float(similarity))


def match_person_reentry(new_features, new_center, history, frame_count):
    """Match person re-entering the scene
    
    Returns: (matched_old_id, similarity_score) or (None, 0.0)
    """
    best_match = None
    best_similarity = PERSON_REID_SIMILARITY_THRESHOLD
    
    for old_id, entry in list(history.items()):
        frames_since_seen = frame_count - entry["last_frame"]
        max_frames = PERSON_OWNER_HISTORY_FRAMES if entry.get("is_owner", False) else PERSON_FEATURE_HISTORY_FRAMES
        
        if frames_since_seen > max_frames or entry["still_tracked"]:
            continue
        
        old_features = entry["features"]
        old_center = entry["last_center"]
        
        # Feature similarity
        feat_sim = compute_feature_similarity(new_features, old_features)
        
        # Spatial similarity
        spatial_dist = math.hypot(new_center[0] - old_center[0], new_center[1] - old_center[1])
        spatial_bonus = max(0.0, 1.0 - (spatial_dist / PERSON_POSITION_REENTRY_THRESHOLD))
        
        # Combined similarity
        combined_sim = feat_sim * 0.9 + spatial_bonus * 0.1
        
        if combined_sim > best_similarity:
            best_similarity = combined_sim
            best_match = old_id
    
    return best_match, best_similarity if best_match else 0.0

"""
Visualization Module - Handle All Drawing Functions
"""
import cv2
from config import LABEL_FONT_SIZE, LABEL_FONT_THICKNESS, DISTANCE_FONT_SIZE, DISTANCE_FONT_THICKNESS
from utils import overlay_mask


def draw_dashed_rectangle(frame, pt1, pt2, color, thickness=2, gap=10):
    """Draw dashed rectangle (for interpolated/memory objects)"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    for x in range(x1, x2, gap * 2):
        cv2.line(frame, (x, y1), (min(x + gap, x2), y1), color, thickness)
    for x in range(x1, x2, gap * 2):
        cv2.line(frame, (x, y2), (min(x + gap, x2), y2), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(frame, (x1, y), (x1, min(y + gap, y2)), color, thickness)
    for y in range(y1, y2, gap * 2):
        cv2.line(frame, (x2, y), (x2, min(y + gap, y2)), color, thickness)


def draw_box_with_label(frame, x1, y1, x2, y2, label, color_bgr, is_interpolated=False):
    """Draw bounding box and label"""
    x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
    
    if is_interpolated:
        draw_dashed_rectangle(frame, (x1i, y1i), (x2i, y2i), color_bgr, 2)
    else:
        cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), color_bgr, 2)

    # Handle multi-line labels
    lines = label.split('\n') if '\n' in label else [label]
    y_offset = 0
    
    for line in lines:
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SIZE, LABEL_FONT_THICKNESS)
        y_top = max(0, y1i - th - 6 - y_offset)
        cv2.rectangle(frame, (x1i, y_top), (x1i + tw + 6, y_top + th + 6), color_bgr, -1)
        cv2.putText(
            frame, line, (x1i + 3, y_top + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SIZE, (255, 255, 255),
            LABEL_FONT_THICKNESS, cv2.LINE_AA
        )
        y_offset += th + 10


def draw_detection(frame, bbox, mask, label, color_bgr, is_interpolated=False):
    """Draw single detection result (box + label + mask)"""
    if mask is not None and not is_interpolated:
        frame = overlay_mask(frame, mask, color_bgr)
    x1, y1, x2, y2 = bbox
    draw_box_with_label(frame, x1, y1, x2, y2, label, color_bgr, is_interpolated)
    return frame


def draw_owner_connection(frame, person_bbox, luggage_bbox, distance):
    """Draw connection line between owner and luggage"""
    px1, py1, px2, py2 = person_bbox
    lx1, ly1, lx2, ly2 = luggage_bbox
    
    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
    lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
    
    p_pt = (int(pcx), int(pcy))
    l_pt = (int(lcx), int(lcy))
    
    cv2.circle(frame, p_pt, 4, (255, 0, 0), -1)
    cv2.circle(frame, l_pt, 4, (0, 0, 255), -1)
    cv2.line(frame, p_pt, l_pt, (0, 255, 255), 2, cv2.LINE_AA)
    
    mid = ((p_pt[0] + l_pt[0]) // 2, (p_pt[1] + l_pt[1]) // 2)
    cv2.putText(
        frame, f"{distance:.1f}px", (mid[0] + 8, mid[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, DISTANCE_FONT_SIZE, (0, 255, 255),
        DISTANCE_FONT_THICKNESS, cv2.LINE_AA
    )
    
    return frame

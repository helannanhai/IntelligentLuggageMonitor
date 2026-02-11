import time
import math
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------
# Configuration
# ----------------------------
MODEL_PATH = "yolov8n-seg.pt"

VIDEO_PATH1 = r"Dataset\JiMeng\2.mp4"
VIDEO_PATH1 = r"Dataset\download.MP4"
VIDEO_PATH1 = r"Dataset\Hailuo_Video_Create a CCTV footage of peopl_398561169233838086.mp4"
VIDEO_PATH1 = r"Dataset\JiMeng\1.mp4"
VIDEO_PATH = r"Dataset\JiMeng\1_1.mov"
VIDEO_PATH1 = r""

# Output settings
SAVE_OUTPUT_VIDEO = 1         # 1=save output video, 0=only display (no save)
OUTPUT_VIDEO_PATH = r"test_video\outputs\output_result.mp4"  # Output video path
OUTPUT_VIDEO_CODEC = "mp4v"   # Video codec: 'mp4v', 'XVID', 'H264', etc.

# COCO class ids
PERSON_CLASS_ID = 0
LUGGAGE_CLASS_IDS = {24, 26, 28}  # backpack, handbag, suitcase

# Inference settings
CONF_THRES = 0.15         # Lower threshold for better continuity (was 0.25)
IOU_THRES = 0.4           # NMS IOU threshold (increased for better overlap handling)
MASK_ALPHA = 0.35
IMGSZ = 640               # Inference image size (larger = better but slower)

# Continuity enhancement settings
MAX_LOST_FRAMES = 30      # Keep tracking objects for N frames after detection loss
POSITION_SMOOTHING = 0.3  # Exponential smoothing factor (0=no smooth, 1=full smooth)

# Text display settings
LABEL_FONT_SIZE = 0.5           # Font size for detection box labels (person/luggage ID)
LABEL_FONT_THICKNESS = 1        # Font thickness for detection box labels
DISTANCE_FONT_SIZE = 0.4        # Font size for distance text between owner and luggage
DISTANCE_FONT_THICKNESS = 1     # Font thickness for distance text

# Owner detection settings (luggage ownership tracking)
OWNER_DISTANCE_THRESHOLD = 350    # Base distance in pixels (for reference person size)
OWNER_DISTANCE_SCALE_FACTOR = 2.0 # Distance = person_height * this_factor (dynamic scaling)
OWNER_CONFIRM_SECONDS = 2.0       # Seconds person must stay near luggage to confirm ownership
ABANDON_CONFIRM_SECONDS = 3.0     # Seconds owner must be away to confirm abandonment
OWNER_HISTORY_FRAMES = 60         # Number of frames to track ownership history
OWNER_CHANGE_THRESHOLD = 0.7      # Ratio of history needed to change owner (70%)
# Runtime-derived frame thresholds (computed after opening video capture)
OWNER_CONFIRM_FRAMES = None
ABANDON_CONFIRM_FRAMES = None


# ---- Small generic helpers (parse bbox, center, unified draw) ----
def parse_box(box, masks_data=None, idx=None):
    """Return (conf, cls_id, track_id, (x1,y1,x2,y2), mask_float)."""
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
    track_id = int(box.id.item()) if box.id is not None else None
    mask_float = None
    if masks_data is not None and idx is not None:
        mask_float = masks_data[idx].detach().cpu().numpy()
    return conf, cls_id, track_id, (x1, y1, x2, y2), mask_float


def center_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def get_unique_filename(filepath):
    """Generate a unique filename by appending numbers if file exists.
    Example: output.mp4 -> output_1.mp4 -> output_2.mp4
    """
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


def draw_detection(frame, bbox, mask, label, color_bgr, mask_alpha=MASK_ALPHA, is_interpolated=False):
    if mask is not None and not is_interpolated:
        # Don't show mask for interpolated objects (we don't have accurate mask)
        frame = overlay_mask(frame, mask, color_bgr, mask_alpha)
    x1, y1, x2, y2 = bbox
    draw_box_with_label(frame, x1, y1, x2, y2, label, color_bgr, is_interpolated)
    return frame


def calculate_dynamic_distance_threshold(person_bbox, luggage_bbox):
    """Calculate dynamic distance threshold based on person size.
    Larger (closer) people get larger thresholds, smaller (farther) people get smaller thresholds.
    """
    px1, py1, px2, py2 = person_bbox
    person_height = py2 - py1
    person_width = px2 - px1
    
    # Use height as primary indicator of depth (closer = taller)
    # Dynamic threshold = person_height * scale_factor
    dynamic_threshold = person_height * OWNER_DISTANCE_SCALE_FACTOR
    
    # Ensure threshold is within reasonable bounds (min 100px, max 600px)
    dynamic_threshold = max(100, min(600, dynamic_threshold))
    
    return dynamic_threshold


def update_luggage_state(state_dict, tid, luggage_bbox, people_data, frame_count):
    """Improved luggage ownership tracking with history and dynamic distance thresholds.
    Returns (state, ownership_text, min_dist, nearest_person_id).
    
    Args:
        people_data: List of (cx, cy, pid, bbox) for each person
    """
    
    # Initialize state for new luggage
    if tid not in state_dict:
        state_dict[tid] = {
            "owner_id": None,              # Current confirmed owner
            "candidate_owner": None,        # Person being considered as owner
            "confirmed": False,             # Whether ownership is confirmed
            "first_seen_frame": frame_count,
            "owner_near_frames": 0,         # Consecutive frames owner stayed near
            "owner_absent_frames": 0,       # Consecutive frames owner is away
            "proximity_history": {},        # {person_id: count} for recent frames
            "history_buffer": [],           # Recent person_ids within threshold
            "last_position": center_from_bbox(luggage_bbox)
        }

    state = state_dict[tid]
    lcx, lcy = center_from_bbox(luggage_bbox)
    
    # Find all people near this luggage (with dynamic thresholds)
    nearby_people = []  # [(person_id, distance, dynamic_threshold), ...]
    min_dist = float('inf')
    nearest_person_id = None
    nearest_person_bbox = None
    
    for pcx, pcy, pid, pbbox in people_data:
        if pid is None:
            continue
        dist = math.hypot(lcx - pcx, lcy - pcy)
        
        # Calculate dynamic threshold based on person size
        dynamic_threshold = calculate_dynamic_distance_threshold(pbbox, luggage_bbox)
        
        if dist < min_dist:
            min_dist = dist
            nearest_person_id = pid
            nearest_person_bbox = pbbox
        
        if dist < dynamic_threshold:
            nearby_people.append((pid, dist, dynamic_threshold))
    
    # Sort by distance (closest first)
    nearby_people.sort(key=lambda x: x[1])
    
    # Update proximity history (sliding window)
    if len(state["history_buffer"]) >= OWNER_HISTORY_FRAMES:
        # Remove oldest entry from counts
        old_pid = state["history_buffer"].pop(0)
        if old_pid in state["proximity_history"]:
            state["proximity_history"][old_pid] -= 1
            if state["proximity_history"][old_pid] <= 0:
                del state["proximity_history"][old_pid]
    
    # Add current frame's nearest person to history
    current_nearest = nearby_people[0][0] if nearby_people else None
    current_threshold = nearby_people[0][2] if nearby_people else OWNER_DISTANCE_THRESHOLD
    state["history_buffer"].append(current_nearest)
    if current_nearest is not None:
        state["proximity_history"][current_nearest] = state["proximity_history"].get(current_nearest, 0) + 1
    
    # === OWNERSHIP LOGIC ===
    current_owner = state["owner_id"]
    
    # Check if current owner is nearby (use dynamic threshold)
    owner_is_near = False
    if current_owner:
        for pid, dist, threshold in nearby_people:
            if pid == current_owner:
                owner_is_near = True
                break
    
    if not state["confirmed"]:
        # PHASE 1: Initial ownership detection
        if current_nearest is not None:
            if state["candidate_owner"] == current_nearest:
                state["owner_near_frames"] += 1
                if state["owner_near_frames"] >= OWNER_CONFIRM_FRAMES:
                    # Confirm ownership after sustained proximity
                    state["owner_id"] = current_nearest
                    state["confirmed"] = True
                    state["owner_absent_frames"] = 0
            else:
                # Different person is closest, reset candidate
                state["candidate_owner"] = current_nearest
                state["owner_near_frames"] = 1
        else:
            # No one nearby, reset
            state["owner_near_frames"] = 0
            state["candidate_owner"] = None
    
    else:
        # PHASE 2: Ownership confirmed - track owner presence
        if owner_is_near:
            # Owner is still nearby
            state["owner_absent_frames"] = 0
        else:
            # Owner is not nearby
            state["owner_absent_frames"] += 1
            
            # Check if we should consider abandonment or transfer
            if state["owner_absent_frames"] >= ABANDON_CONFIRM_FRAMES:
                # Owner has been away long enough
                if current_nearest is not None:
                    # Someone else is consistently near - consider transfer
                    # Check if new person has dominated recent history
                    if current_nearest in state["proximity_history"]:
                        history_ratio = state["proximity_history"][current_nearest] / len(state["history_buffer"])
                        if history_ratio >= OWNER_CHANGE_THRESHOLD:
                            # Transfer ownership to new person
                            state["owner_id"] = current_nearest
                            state["owner_absent_frames"] = 0
                            state["proximity_history"] = {current_nearest: state["proximity_history"][current_nearest]}
                else:
                    # No one nearby - mark as abandoned
                    state["confirmed"] = False
                    state["owner_id"] = None
                    state["candidate_owner"] = None
                    state["owner_near_frames"] = 0
    
    # Generate ownership text
    if state["confirmed"] and state["owner_id"] is not None:
        if state["owner_absent_frames"] > 0:
            ownership_text = f"OWNER:{state['owner_id']} (away {state['owner_absent_frames']}f)"
        else:
            ownership_text = f"OWNER:{state['owner_id']}"
    elif state["candidate_owner"] is not None:
        ownership_text = f"DETECTING... ({state['owner_near_frames']}/{OWNER_CONFIRM_FRAMES}f)"
    else:
        ownership_text = "ABANDONED"
    
    return state, ownership_text, min_dist, nearest_person_id

# ----------------------------
# Detection continuity helpers
# ----------------------------
def smooth_bbox(current_bbox, previous_bbox, alpha=POSITION_SMOOTHING):
    """Apply exponential smoothing to bbox coordinates for stability."""
    if previous_bbox is None:
        return current_bbox
    
    x1, y1, x2, y2 = current_bbox
    px1, py1, px2, py2 = previous_bbox
    
    # Smooth each coordinate
    sx1 = alpha * x1 + (1 - alpha) * px1
    sy1 = alpha * y1 + (1 - alpha) * py1
    sx2 = alpha * x2 + (1 - alpha) * px2
    sy2 = alpha * y2 + (1 - alpha) * py2
    
    return (sx1, sy1, sx2, sy2)

def update_object_memory(memory_dict, detections, frame_count, obj_type="person"):
    """Update object memory with current detections and handle lost objects.
    Returns: updated list of detections including interpolated lost objects."""
    
    # Update memory with current detections
    detected_ids = set()
    for det in detections:
        conf, cls_id, tid, bbox, mask = det
        if tid is not None:
            detected_ids.add(tid)
            if tid not in memory_dict:
                memory_dict[tid] = {
                    "bbox": bbox,
                    "last_seen": frame_count,
                    "lost_count": 0,
                    "conf": conf,
                    "mask": mask,
                    "cls_id": cls_id
                }
            else:
                # Smooth the bbox
                smoothed_bbox = smooth_bbox(bbox, memory_dict[tid]["bbox"])
                memory_dict[tid].update({
                    "bbox": smoothed_bbox,
                    "last_seen": frame_count,
                    "lost_count": 0,
                    "conf": conf,
                    "mask": mask
                })
    
    # Handle lost objects (not detected in current frame)
    result_detections = list(detections)
    lost_ids = []
    
    for tid, mem in list(memory_dict.items()):
        if tid not in detected_ids:
            mem["lost_count"] += 1
            
            if mem["lost_count"] <= MAX_LOST_FRAMES:
                # Keep showing the object at last known position
                # Slightly lower confidence to indicate uncertainty
                interpolated_conf = mem["conf"] * (1.0 - mem["lost_count"] / (MAX_LOST_FRAMES * 2))
                result_detections.append((
                    interpolated_conf,
                    mem["cls_id"],
                    tid,
                    mem["bbox"],
                    mem["mask"]
                ))
            else:
                # Object truly lost, remove from memory
                lost_ids.append(tid)
    
    # Clean up completely lost objects
    for tid in lost_ids:
        del memory_dict[tid]
    
    return result_detections

# ----------------------------
# Drawing helpers
# ----------------------------
def draw_dashed_rectangle(frame, pt1, pt2, color, thickness=2, gap=10):
    """Draw a dashed rectangle to indicate interpolated/memory objects."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Top line
    for x in range(x1, x2, gap * 2):
        cv2.line(frame, (x, y1), (min(x + gap, x2), y1), color, thickness)
    # Bottom line
    for x in range(x1, x2, gap * 2):
        cv2.line(frame, (x, y2), (min(x + gap, x2), y2), color, thickness)
    # Left line
    for y in range(y1, y2, gap * 2):
        cv2.line(frame, (x1, y), (x1, min(y + gap, y2)), color, thickness)
    # Right line
    for y in range(y1, y2, gap * 2):
        cv2.line(frame, (x2, y), (x2, min(y + gap, y2)), color, thickness)

def draw_box_with_label(frame, x1, y1, x2, y2, label, color_bgr, is_interpolated=False):
    # Draw bounding box (dashed if interpolated)
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
            frame,
            line,
            (x1i + 3, y_top + th + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SIZE,
            (255, 255, 255),
            LABEL_FONT_THICKNESS,
            cv2.LINE_AA,
        )
        y_offset += th + 10

def overlay_mask(frame_bgr, mask_float, color_bgr, alpha):
    # Convert mask (float) to uint8, then resize to frame size if needed
    h, w = frame_bgr.shape[:2]
    mask_u8 = (mask_float > 0.5).astype(np.uint8) * 255
    if mask_u8.shape[0] != h or mask_u8.shape[1] != w:
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    # Alpha blend on masked region
    overlay = frame_bgr.copy()
    overlay[mask_u8 > 0] = color_bgr
    return cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0.0)

# ----------------------------
# Selection helper
# ----------------------------
def pick_best_instance(res, wanted_classes):
    # Pick exactly one instance among wanted classes:
    # 1) highest confidence
    # 2) if tie, largest bbox area
    if res.boxes is None or len(res.boxes) == 0:
        return None

    masks_data = res.masks.data if res.masks is not None else None
    best = None  # (conf, area, cls_id, track_id, bbox, mask_float)

    for i, b in enumerate(res.boxes):
        conf, cls_id, track_id, bbox, mask_float = parse_box(b, masks_data, i)
        if cls_id not in wanted_classes:
            continue
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        cand = (conf, area, cls_id, track_id, bbox, mask_float)
        if best is None or (conf > best[0]) or (abs(conf - best[0]) < 1e-6 and area > best[1]):
            best = cand

    return best

def pick_best_luggage(res, luggage_classes, person_bbox=None):
    """Pick best luggage, avoiding person occlusion. If luggage overlaps with person, 
    try to pick the one with highest confidence despite overlap."""
    if res.boxes is None or len(res.boxes) == 0:
        return None

    masks_data = res.masks.data if res.masks is not None else None
    candidates = []

    for i, b in enumerate(res.boxes):
        conf, cls_id, track_id, bbox, mask_float = parse_box(b, masks_data, i)
        if cls_id not in luggage_classes:
            continue
        x1, y1, x2, y2 = bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        overlap = 0.0
        if person_bbox is not None:
            px1, py1, px2, py2 = person_bbox
            inter_x1, inter_y1 = max(x1, px1), max(y1, py1)
            inter_x2, inter_y2 = min(x2, px2), min(y2, py2)
            if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union_area = area + (px2 - px1) * (py2 - py1) - inter_area
                overlap = inter_area / union_area if union_area > 0 else 0

        candidates.append((conf, area, cls_id, track_id, bbox, mask_float, overlap))

    if not candidates:
        return None

    best = max(candidates, key=lambda c: c[0])
    return best[:6]

# ----------------------------
# Main
# ----------------------------
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

# Compute frame thresholds from configured seconds using video FPS
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
OWNER_CONFIRM_FRAMES = max(1, int(round(OWNER_CONFIRM_SECONDS * fps)))
ABANDON_CONFIRM_FRAMES = max(1, int(round(ABANDON_CONFIRM_SECONDS * fps)))
print(f"[v3.0 Improved Tracking] FPS={fps:.1f}")
print(f"  Owner confirm: {OWNER_CONFIRM_SECONDS}s -> {OWNER_CONFIRM_FRAMES} frames")
print(f"  Abandon confirm: {ABANDON_CONFIRM_SECONDS}s -> {ABANDON_CONFIRM_FRAMES} frames")
print(f"  History window: {OWNER_HISTORY_FRAMES} frames ({OWNER_HISTORY_FRAMES/fps:.1f}s)")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup output video writer if enabled
out = None
final_output_path = OUTPUT_VIDEO_PATH  # Track the actual path used
if SAVE_OUTPUT_VIDEO:
    # Get unique output filename to avoid overwriting
    final_output_path = get_unique_filename(OUTPUT_VIDEO_PATH)
    if final_output_path != OUTPUT_VIDEO_PATH:
        print(f"[Output] File exists, using new name: {os.path.basename(final_output_path)}")
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_VIDEO_CODEC)
    out = cv2.VideoWriter(final_output_path, fourcc, fps, (frame_width, frame_height))
    
    if out.isOpened():
        print(f"[Output] Saving video to: {final_output_path}")
        print(f"  Resolution: {frame_width}x{frame_height}, FPS: {fps:.1f}, Codec: {OUTPUT_VIDEO_CODEC}")
    else:
        print(f"[Warning] Failed to create output video. Will only display.")
        out = None

# Create a resizable window and adjust to current video resolution
cv2.namedWindow("view", cv2.WINDOW_NORMAL)
cv2.resizeWindow("view", frame_width, frame_height)

# Sync playback speed using video timestamps (prevents running faster than the original)
start_wall = time.perf_counter()
start_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

frame_count = 0

# Only keep person + luggage classes
classes_keep = [PERSON_CLASS_ID, *sorted(list(LUGGAGE_CLASS_IDS))]

# Luggage owner tracking state
luggage_ownership_state = {}

# Object memory for continuity (stores last known positions)
people_memory = {}    # {person_id: {bbox, last_seen, lost_count, ...}}
luggage_memory = {}   # {luggage_id: {bbox, last_seen, lost_count, ...}}

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame_count += 1

    # Timestamp-based playback sync
    cur_msec = cap.get(cv2.CAP_PROP_POS_MSEC) - start_msec
    target_sec = max(0.0, float(cur_msec) / 1000.0)
    while (time.perf_counter() - start_wall) < target_sec:
        time.sleep(0.001)

    # Tracking for stable unique IDs across frames
    # Lower iou threshold to keep overlapping detections (person + luggage)
    res = model.track(
        frame,
        persist=True,
        tracker="trackers/botsort_luggage.yaml",
        classes=classes_keep,
        conf=CONF_THRES,
        iou=IOU_THRES,      # Lower NMS IOU to preserve overlapping boxes
        imgsz=IMGSZ,        # Use larger inference size for better detection
        verbose=False
    )[0]

    # Collect ALL people and ALL luggage
    all_people = []  # [(conf, cls_id, tid, bbox, mask), ...]
    all_luggage = []  # [(conf, cls_id, tid, bbox, mask), ...]
    
    if res.boxes is not None:
        masks_data = res.masks.data if res.masks is not None else None
        for i, b in enumerate(res.boxes):
            conf, cls_id, tid, bbox, mask = parse_box(b, masks_data, i)
            if cls_id == PERSON_CLASS_ID:
                all_people.append((conf, cls_id, tid, bbox, mask))
            elif cls_id in LUGGAGE_CLASS_IDS:
                all_luggage.append((conf, cls_id, tid, bbox, mask))
    
    # Apply object memory for detection continuity
    all_people = update_object_memory(people_memory, all_people, frame_count, "person")
    all_luggage = update_object_memory(luggage_memory, all_luggage, frame_count, "luggage")
    
    # Rebuild people data with bbox after memory update
    all_people_data = []  # [(cx, cy, tid, bbox), ...]
    for conf, cls_id, tid, bbox, mask in all_people:
        if tid is not None:
            cx, cy = center_from_bbox(bbox)
            all_people_data.append((cx, cy, tid, bbox))
    
    # Debug: Print detection info every 30 frames
    if frame_count % 30 == 0:
        people_mem_count = len([1 for m in people_memory.values() if m["lost_count"] > 0])
        luggage_mem_count = len([1 for m in luggage_memory.values() if m["lost_count"] > 0])
        print(f"[Frame {frame_count}] Detected: {len(all_people)} people ({people_mem_count} memory), {len(all_luggage)} luggage ({luggage_mem_count} memory)")

    # Draw all people
    for conf, cls_id, tid, bbox, mask in all_people:
        # Check if this person is from memory (interpolated)
        is_interpolated = tid in people_memory and people_memory[tid]["lost_count"] > 0
        label = f"person id={tid} {conf:.2f}" if tid is not None else f"person {conf:.2f}"
        if is_interpolated:
            label += " [MEM]"
        frame = draw_detection(frame, bbox, mask, label, (255, 0, 0), is_interpolated=is_interpolated)

    # Process and draw all luggage with ownership tracking
    luggage_bindings = []  # [(luggage_info, owner_id, distance), ...]
    
    for conf, cls_id, tid, bbox, mask in all_luggage:
        if tid is None:
            ownership_text = "NO_ID"
            owner_id = None
        else:
            state, ownership_text, min_dist, nearest_pid = update_luggage_state(
                luggage_ownership_state, tid, bbox, all_people_data, frame_count
            )
            owner_id = state.get('owner_id')
            
            # Store binding for drawing lines later
            if owner_id is not None:
                luggage_bindings.append(((tid, bbox), owner_id, min_dist))
            
            # Debug output every 30 frames
            if frame_count % 30 == 0:
                dist_str = "N/A" if min_dist == float('inf') else f"{int(min_dist)}px"
                confirmed = "✓" if state["confirmed"] else "?"
                absent = state.get("owner_absent_frames", 0)
                history_len = len(state.get("proximity_history", {}))
                
                # Calculate dynamic threshold for owner (for debugging)
                debug_threshold = "N/A"
                if owner_id is not None:
                    for cx, cy, pid, pbbox in all_people_data:
                        if pid == owner_id:
                            debug_threshold = f"{int(calculate_dynamic_distance_threshold(pbbox, bbox))}px"
                            break
                
                print(f"  Luggage {tid}: owner={owner_id}{confirmed}, min_dist={dist_str}, threshold={debug_threshold}, absent={absent}f, history={history_len} people")

        # Check if this luggage is from memory (interpolated)
        is_interpolated = tid is not None and tid in luggage_memory and luggage_memory[tid]["lost_count"] > 0
        label_text = f"luggage id={tid} {conf:.2f}\n{ownership_text}" if tid is not None else f"luggage {conf:.2f}\nNO_ID"
        if is_interpolated:
            label_text = f"luggage id={tid} {conf:.2f} [MEM]\n{ownership_text}"
        
        frame = draw_detection(frame, bbox, mask, label_text, (0, 0, 255), is_interpolated=is_interpolated)

    # Draw lines between luggage and their owners
    for (luggage_tid, luggage_bbox), owner_id, distance in luggage_bindings:
        # Find owner bbox
        owner_bbox = None
        for conf, cls_id, tid, bbox, mask in all_people:
            if tid == owner_id:
                owner_bbox = bbox
                break
        
        if owner_bbox is not None:
            # Calculate centers
            lx1, ly1, lx2, ly2 = luggage_bbox
            px1, py1, px2, py2 = owner_bbox
            
            lcx, lcy = (lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0
            pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
            
            l_pt = (int(lcx), int(lcy))
            p_pt = (int(pcx), int(pcy))
            
            # Draw connection
            cv2.circle(frame, p_pt, 4, (255, 0, 0), -1)
            cv2.circle(frame, l_pt, 4, (0, 0, 255), -1)
            cv2.line(frame, p_pt, l_pt, (0, 255, 255), 2, cv2.LINE_AA)
            
            # Draw distance
            mid = ((p_pt[0] + l_pt[0]) // 2, (p_pt[1] + l_pt[1]) // 2)
            cv2.putText(
                frame,
                f"{distance:.1f}px",
                (mid[0] + 8, mid[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                DISTANCE_FONT_SIZE,
                (0, 255, 255),
                DISTANCE_FONT_THICKNESS,
                cv2.LINE_AA,
            )

    # Show frame
    cv2.imshow("view", frame)
    
    # Write frame to output video if enabled
    if out is not None and out.isOpened():
        out.write(frame)

    # Quit
    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
if out is not None:
    out.release()
    print(f"\n[Output] Video saved successfully to: {final_output_path}")
cv2.destroyAllWindows()

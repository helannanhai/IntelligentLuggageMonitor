"""
Luggage Ownership Tracking Module
"""
import math
from utils import center_from_bbox
import config


def calculate_dynamic_distance_threshold(person_bbox, luggage_bbox):
    """Calculate dynamic distance threshold based on person size"""
    px1, py1, px2, py2 = person_bbox
    person_height = py2 - py1
    dynamic_threshold = person_height * config.OWNER_DISTANCE_SCALE_FACTOR
    return max(100, min(600, dynamic_threshold))


def update_luggage_state(state_dict, tid, luggage_bbox, people_data, frame_count, person_id_mapping):
    """Update luggage ownership state
    
    Returns: (state, ownership_text, min_dist, nearest_person_id)
    """
    # Initialize new luggage state
    if tid not in state_dict:
        state_dict[tid] = {
            "owner_id": None,
            "original_owner_id": None,
            "candidate_owner": None,
            "confirmed": False,
            "first_seen_frame": frame_count,
            "owner_near_frames": 0,
            "owner_absent_frames": 0,
            "proximity_history": {},
            "history_buffer": [],
            "last_position": center_from_bbox(luggage_bbox),
            "stranger_detected": False,
            "stranger_id": None
        }

    state = state_dict[tid]
    lcx, lcy = center_from_bbox(luggage_bbox)
    
    # Find nearby people
    nearby_people = []
    min_dist = float('inf')
    nearest_person_id = None
    nearest_person_bbox = None
    
    for pcx, pcy, pid, pbbox in people_data:
        if pid is None:
            continue
        dist = math.hypot(lcx - pcx, lcy - pcy)
        dynamic_threshold = calculate_dynamic_distance_threshold(pbbox, luggage_bbox)
        
        if dist < min_dist:
            min_dist = dist
            nearest_person_id = pid
            nearest_person_bbox = pbbox
        
        if dist < dynamic_threshold:
            nearby_people.append((pid, dist, dynamic_threshold))
    
    nearby_people.sort(key=lambda x: x[1])
    
    # Update proximity history (sliding window)
    if len(state["history_buffer"]) >= config.OWNER_HISTORY_FRAMES:
        old_pid = state["history_buffer"].pop(0)
        if old_pid in state["proximity_history"]:
            state["proximity_history"][old_pid] -= 1
            if state["proximity_history"][old_pid] <= 0:
                del state["proximity_history"][old_pid]
    
    current_nearest = nearby_people[0][0] if nearby_people else None
    state["history_buffer"].append(current_nearest)
    if current_nearest is not None:
        state["proximity_history"][current_nearest] = state["proximity_history"].get(current_nearest, 0) + 1
    
    # Ownership logic
    current_owner = state["owner_id"]
    owner_is_near = False
    if current_owner:
        for pid, dist, threshold in nearby_people:
            if pid == current_owner:
                owner_is_near = True
                break
    
    if not state["confirmed"]:
        # Stage 1: Initial ownership detection
        if current_nearest is not None:
            if state["original_owner_id"] is not None:
                # Luggage was previously abandoned
                if current_nearest == state["original_owner_id"]:
                    # Original owner reclaiming
                    state["owner_id"] = current_nearest
                    state["confirmed"] = True
                    state["owner_absent_frames"] = 0
                    state["stranger_detected"] = False
                else:
                    # Stranger taking luggage
                    if state["candidate_owner"] == current_nearest:
                        state["owner_near_frames"] += 1
                        if state["owner_near_frames"] >= config.OWNER_CONFIRM_FRAMES:
                            state["stranger_detected"] = True
                            state["stranger_id"] = current_nearest
                            state["owner_near_frames"] = 0
                    else:
                        state["candidate_owner"] = current_nearest
                        state["owner_near_frames"] = 1
            else:
                # First detection - normal binding
                if state["candidate_owner"] == current_nearest:
                    state["owner_near_frames"] += 1
                    if state["owner_near_frames"] >= config.OWNER_CONFIRM_FRAMES:
                        state["owner_id"] = current_nearest
                        state["original_owner_id"] = current_nearest
                        state["confirmed"] = True
                        state["owner_absent_frames"] = 0
                else:
                    state["candidate_owner"] = current_nearest
                    state["owner_near_frames"] = 1
        else:
            state["owner_near_frames"] = 0
            state["candidate_owner"] = None
    else:
        # Stage 2: Ownership confirmed - track owner status
        if owner_is_near:
            state["owner_absent_frames"] = 0
        else:
            state["owner_absent_frames"] += 1
            
            if state["owner_absent_frames"] >= config.ABANDON_CONFIRM_FRAMES:
                if current_nearest is not None:
                    if current_nearest in state["proximity_history"]:
                        history_ratio = state["proximity_history"][current_nearest] / len(state["history_buffer"])
                        if history_ratio >= config.OWNER_CHANGE_THRESHOLD:
                            if current_nearest == state["original_owner_id"]:
                                # Original owner returned
                                state["owner_id"] = current_nearest
                                state["owner_absent_frames"] = 0
                                state["stranger_detected"] = False
                                state["stranger_id"] = None
                                state["proximity_history"] = {current_nearest: state["proximity_history"][current_nearest]}
                            else:
                                # Stranger stealing luggage
                                state["stranger_detected"] = True
                                state["stranger_id"] = current_nearest
                else:
                    # 无人在附近 - 标记为遗弃
                    if not state["stranger_detected"]:
                        state["confirmed"] = False
                        state["owner_id"] = None
                        state["candidate_owner"] = None
                        state["owner_near_frames"] = 0
    
    # 生成所有权文本
    if state["stranger_detected"]:
        ownership_text = f"[STOLEN BY ID:{state['stranger_id']}] (was OWNER:{state['original_owner_id']})"
    elif state["confirmed"] and state["owner_id"] is not None:
        if state["owner_absent_frames"] > 0:
            ownership_text = f"OWNER:{state['owner_id']} (away {state['owner_absent_frames']}f)"
        else:
            ownership_text = f"OWNER:{state['owner_id']}"
    elif state["candidate_owner"] is not None:
        ownership_text = f"DETECTING... ({state['owner_near_frames']}/{config.OWNER_CONFIRM_FRAMES}f)"
    else:
        ownership_text = "ABANDONED"
    
    return state, ownership_text, min_dist, nearest_person_id

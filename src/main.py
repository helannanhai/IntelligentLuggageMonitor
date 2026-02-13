"""
Intelligent Luggage Monitoring System - Main Entry (Modular Version)
Usage: python main.py
"""
import time
import cv2
from ultralytics import YOLO

import config
from utils import parse_box, center_from_bbox, get_unique_filename, smooth_bbox
from visualizer import draw_detection, draw_owner_connection
from reid import extract_person_features, match_person_reentry
from ownership import update_luggage_state


class LuggageMonitor:
    """Main luggage monitoring class"""
    
    def __init__(self, model):
        self.model = model
        self.cap = None
        self.out = None
        self.final_output_path = config.OUTPUT_VIDEO_PATH
        self.fps = 0.0
        self.frame_width = 0
        self.frame_height = 0
        self.start_wall = 0.0
        self.start_msec = 0.0
        self.frame_count = 0
        
        # Tracking state
        self.people_memory = {}
        self.luggage_memory = {}
        self.luggage_ownership_state = {}
        
        # Re-ID state
        self.people_features_history = {}
        self.person_id_mapping = {}
        self.frame_data = None
        
        # Consecutive frame confirmation (filter background false positives)
        self.person_consecutive_frames = {}
        self.luggage_consecutive_frames = {}

    def init_video_io(self, video_path):
        """Initialize video input/output"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        config.OWNER_CONFIRM_FRAMES = max(1, int(round(config.OWNER_CONFIRM_SECONDS * self.fps)))
        config.ABANDON_CONFIRM_FRAMES = max(1, int(round(config.ABANDON_CONFIRM_SECONDS * self.fps)))
        
        print(f"[Init] FPS={self.fps:.1f}")
        print(f"  Owner confirm: {config.OWNER_CONFIRM_SECONDS}s -> {config.OWNER_CONFIRM_FRAMES} frames")
        print(f"  Abandon confirm: {config.ABANDON_CONFIRM_SECONDS}s -> {config.ABANDON_CONFIRM_FRAMES} frames")
        print(f"  Consecutive frame threshold: {config.MIN_CONSECUTIVE_FRAMES} frames")

        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if config.SAVE_OUTPUT_VIDEO:
            self.final_output_path = get_unique_filename(config.OUTPUT_VIDEO_PATH)
            fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
            self.out = cv2.VideoWriter(self.final_output_path, fourcc, self.fps, 
                                      (self.frame_width, self.frame_height))
            if self.out.isOpened():
                print(f"[Output] Saving to: {self.final_output_path}")

        cv2.namedWindow("view", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("view", self.frame_width, self.frame_height)

        self.start_wall = time.perf_counter()
        self.start_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        self.cap = cap

    def run_inference(self, frame):
        """Run YOLO detection and tracking"""
        classes_keep = [config.PERSON_CLASS_ID, *sorted(list(config.LUGGAGE_CLASS_IDS))]
        res = self.model.track(
            frame, persist=True, tracker="trackers/botsort_luggage.yaml",
            classes=classes_keep, conf=config.CONF_THRES, iou=config.IOU_THRES,
            imgsz=config.IMGSZ, verbose=False
        )[0]

        all_people = []
        all_luggage = []
        current_person_ids = set()
        current_luggage_ids = set()

        if res.boxes is not None:
            masks_data = res.masks.data if res.masks is not None else None
            for i, b in enumerate(res.boxes):
                conf, cls_id, tid, bbox, mask = parse_box(b, masks_data, i)
                if cls_id == config.PERSON_CLASS_ID:
                    all_people.append((conf, cls_id, tid, bbox, mask))
                    if tid is not None:
                        current_person_ids.add(tid)
                elif cls_id in config.LUGGAGE_CLASS_IDS:
                    all_luggage.append((conf, cls_id, tid, bbox, mask))
                    if tid is not None:
                        current_luggage_ids.add(tid)
        
        # Consecutive frame filtering
        all_people = self._filter_by_consecutive_frames(
            all_people, current_person_ids, self.person_consecutive_frames, "Person")
        all_luggage = self._filter_by_consecutive_frames(
            all_luggage, current_luggage_ids, self.luggage_consecutive_frames, "Luggage")

        return all_people, all_luggage
    
    def _filter_by_consecutive_frames(self, detections, current_ids, counter_dict, obj_type):
        """Filter: only keep objects appearing consecutively for >=MIN_CONSECUTIVE_FRAMES frames"""
        for tid in list(counter_dict.keys()):
            if tid in current_ids:
                counter_dict[tid] += 1
            else:
                del counter_dict[tid]
        
        for tid in current_ids:
            if tid not in counter_dict:
                counter_dict[tid] = 1
        
        filtered = []
        for det in detections:
            conf, cls_id, tid, bbox, mask = det
            if tid is None or counter_dict.get(tid, 0) >= config.MIN_CONSECUTIVE_FRAMES:
                filtered.append(det)
        
        return filtered

    def update_person_features_and_mapping(self, all_people):
        """Update person features and handle Re-ID"""
        for old_id in self.people_features_history:
            self.people_features_history[old_id]["still_tracked"] = False
        
        current_person_ids = set()
        updated_people = []
        
        for conf, cls_id, tid, bbox, mask in all_people:
            if tid is None or cls_id != config.PERSON_CLASS_ID:
                updated_people.append((conf, cls_id, tid, bbox, mask))
                continue
            
            current_person_ids.add(tid)
            new_features = extract_person_features(self.frame_data, bbox)
            new_center = center_from_bbox(bbox)
            
            if tid not in self.people_features_history and tid not in self.person_id_mapping:
                matched_old_id, similarity = match_person_reentry(
                    new_features, new_center, self.people_features_history, self.frame_count)
                
                if matched_old_id is not None:
                    self.person_id_mapping[tid] = matched_old_id
                    print(f"[Re-ID] ID {tid} -> {matched_old_id} (similarity={similarity:.3f})"))
                    self.people_features_history[matched_old_id].update({
                        "features": new_features, "last_frame": self.frame_count,
                        "last_center": new_center, "still_tracked": True
                    })
                else:
                    self.people_features_history[tid] = {
                        "features": new_features, "last_frame": self.frame_count,
                        "last_center": new_center, "still_tracked": True, "is_owner": False
                    }
            elif tid in self.person_id_mapping:
                original_id = self.person_id_mapping[tid]
                self.people_features_history[original_id].update({
                    "features": new_features, "last_frame": self.frame_count,
                    "last_center": new_center, "still_tracked": True
                })
            else:
                self.people_features_history[tid].update({
                    "features": new_features, "last_frame": self.frame_count,
                    "last_center": new_center, "still_tracked": True
                })
            
            updated_people.append((conf, cls_id, tid, bbox, mask))
        
        # Clean up expired entries
        expired_ids = []
        for old_id, entry in self.people_features_history.items():
            max_age = config.PERSON_OWNER_HISTORY_FRAMES * 2 if entry.get("is_owner", False) else config.PERSON_FEATURE_HISTORY_FRAMES * 2
            if self.frame_count - entry["last_frame"] > max_age:
                expired_ids.append(old_id)
        
        for old_id in expired_ids:
            del self.people_features_history[old_id]
            self.person_id_mapping = {k: v for k, v in self.person_id_mapping.items() if v != old_id}
        
        return updated_people

    def apply_memory(self, all_people, all_luggage):
        """Apply memory smoothing"""
        all_people = self._update_object_memory(self.people_memory, all_people)
        all_luggage = self._update_object_memory(self.luggage_memory, all_luggage)

        all_people_data = []
        people_by_id = {}
        for conf, cls_id, tid, bbox, mask in all_people:
            if tid is not None:
                cx, cy = center_from_bbox(bbox)
                original_id = self.person_id_mapping.get(tid, tid)
                all_people_data.append((cx, cy, original_id, bbox))
                people_by_id[original_id] = bbox

        return all_people, all_luggage, all_people_data, people_by_id
    
    def _update_object_memory(self, memory_dict, detections):
        """Update object memory, handle brief loss"""
        detected_ids = set()
        for det in detections:
            conf, cls_id, tid, bbox, mask = det
            if tid is not None:
                detected_ids.add(tid)
                if tid not in memory_dict:
                    memory_dict[tid] = {
                        "bbox": bbox, "last_seen": self.frame_count,
                        "lost_count": 0, "conf": conf, "mask": mask, "cls_id": cls_id
                    }
                else:
                    smoothed_bbox = smooth_bbox(bbox, memory_dict[tid]["bbox"])
                    memory_dict[tid].update({
                        "bbox": smoothed_bbox, "last_seen": self.frame_count,
                        "lost_count": 0, "conf": conf, "mask": mask
                    })
        
        result_detections = list(detections)
        lost_ids = []
        
        for tid, mem in list(memory_dict.items()):
            if tid not in detected_ids:
                mem["lost_count"] += 1
                if mem["lost_count"] <= config.MAX_LOST_FRAMES:
                    interpolated_conf = mem["conf"] * (1.0 - mem["lost_count"] / (config.MAX_LOST_FRAMES * 2))
                    result_detections.append((interpolated_conf, mem["cls_id"], tid, mem["bbox"], mem["mask"]))
                else:
                    lost_ids.append(tid)
        
        for tid in lost_ids:
            del memory_dict[tid]
        
        return result_detections

    def render_frame(self, frame, all_people, all_luggage, all_people_data, people_by_id):
        """Render all detection results to frame"""
        # 绘制人物
        for conf, cls_id, tid, bbox, mask in all_people:
            is_interpolated = tid in self.people_memory and self.people_memory[tid]["lost_count"] > 0
            original_id = self.person_id_mapping.get(tid, tid)
            display_label = f"[ReID:{original_id}]" if tid in self.person_id_mapping else ""
            label = f"person id={tid if tid not in self.person_id_mapping else f'{original_id}(new)'} {conf:.2f}" if tid is not None else f"person {conf:.2f}"
            if is_interpolated:
                label += " [MEM]"
            if display_label:
                label = display_label + " " + label
            frame = draw_detection(frame, bbox, mask, label, (255, 0, 0), is_interpolated)

        # Draw luggage and ownership connections
        for conf, cls_id, tid, bbox, mask in all_luggage:
            if tid is None:
                ownership_text = "NO_ID"
                owner_id = None
                min_dist = float("inf")
            else:
                state, ownership_text, min_dist, nearest_pid = update_luggage_state(
                    self.luggage_ownership_state, tid, bbox, all_people_data, 
                    self.frame_count, self.person_id_mapping)
                owner_id = state.get("owner_id")

            is_interpolated = tid is not None and tid in self.luggage_memory and self.luggage_memory[tid]["lost_count"] > 0
            label_text = f"luggage id={tid} {conf:.2f}\n{ownership_text}" if tid is not None else f"luggage {conf:.2f}\nNO_ID"
            if is_interpolated:
                label_text = f"luggage id={tid} {conf:.2f} [MEM]\n{ownership_text}"

            frame = draw_detection(frame, bbox, mask, label_text, (0, 0, 255), is_interpolated)

            if owner_id is not None and owner_id in people_by_id:
                frame = draw_owner_connection(frame, people_by_id[owner_id], bbox, min_dist)

        return frame

    def run(self, video_path):
        """Main run loop"""
        self.init_video_io(video_path)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            self.frame_count += 1

            # Synchronize playback
            cur_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC) - self.start_msec
            target_sec = max(0.0, float(cur_msec) / 1000.0)
            while (time.perf_counter() - self.start_wall) < target_sec:
                time.sleep(0.001)

            self.frame_data = frame
            all_people, all_luggage = self.run_inference(frame)
            all_people = self.update_person_features_and_mapping(all_people)
            all_people, all_luggage, all_people_data, people_by_id = self.apply_memory(all_people, all_luggage)

            frame = self.render_frame(frame, all_people, all_luggage, all_people_data, people_by_id)

            # Periodic debug output
            if self.frame_count % 30 == 0:
                print(f"[Frame {self.frame_count}] People:{len(all_people)} Luggage:{len(all_luggage)}")

            cv2.imshow("view", frame)
            if self.out is not None and self.out.isOpened():
                self.out.write(frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        self.cap.release()
        if self.out is not None:
            self.out.release()
            print(f"\n[Done] Video saved: {self.final_output_path}")
        cv2.destroyAllWindows()


def main():
    """Program entry point"""
    print("=" * 60)
    print("Intelligent Luggage Monitoring System v3.0 (Modular Version)")
    print("=" * 60)
    
    model = YOLO(config.MODEL_PATH)
    monitor = LuggageMonitor(model)
    monitor.run(config.VIDEO_PATH)


if __name__ == "__main__":
    main()

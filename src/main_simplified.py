"""
Intelligent Luggage Monitoring System - Simplified Version (Same functionality, cleaner code)
Usage: python main_simplified.py
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
        self.frame_count = 0
        
        # Unified tracking state
        self.memory = {'people': {}, 'luggage': {}}  # Merged memory dicts
        self.consecutive = {'people': {}, 'luggage': {}}  # Merged consecutive frame counters
        self.luggage_ownership_state = {}
        
        # Re-ID状态
        self.people_features = {}
        self.person_id_mapping = {}
        self.frame_data = None

    def init_video_io(self, video_path):
        """Initialize video input/output"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        config.OWNER_CONFIRM_FRAMES = max(1, int(round(config.OWNER_CONFIRM_SECONDS * self.fps)))
        config.ABANDON_CONFIRM_FRAMES = max(1, int(round(config.ABANDON_CONFIRM_SECONDS * self.fps)))
        
        print(f"[Init] FPS={self.fps:.1f}, Owner confirm={config.OWNER_CONFIRM_FRAMES}frames, "
              f"Abandon confirm={config.ABANDON_CONFIRM_FRAMES}frames, Consecutive threshold={config.MIN_CONSECUTIVE_FRAMES}frames")

        if config.SAVE_OUTPUT_VIDEO:
            self.final_output_path = get_unique_filename(config.OUTPUT_VIDEO_PATH)
            fourcc = cv2.VideoWriter_fourcc(*config.OUTPUT_VIDEO_CODEC)
            w, h = int(self.cap.get(3)), int(self.cap.get(4))
            self.out = cv2.VideoWriter(self.final_output_path, fourcc, self.fps, (w, h))
            if self.out.isOpened():
                print(f"[Output] Saving to: {self.final_output_path}")

        cv2.namedWindow("view", cv2.WINDOW_NORMAL)
        self.start_wall = time.perf_counter()
        self.start_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)

    def process_frame(self, frame):
        """Process single frame - merge all detection, tracking, Re-ID logic"""
        self.frame_data = frame
        
        # 1. YOLO detection
        res = self.model.track(
            frame, persist=True, tracker="trackers/botsort_luggage.yaml",
            classes=[config.PERSON_CLASS_ID, *config.LUGGAGE_CLASS_IDS],
            conf=config.CONF_THRES, iou=config.IOU_THRES, imgsz=config.IMGSZ, verbose=False
        )[0]

        # 2. Parse detection results and apply consecutive frame filtering + memory smoothing (one pass)
        people, luggage = [], []
        current_ids = {'people': set(), 'luggage': set()}
        
        if res.boxes is not None:
            masks = res.masks.data if res.masks is not None else None
            for i, b in enumerate(res.boxes):
                conf, cls, tid, bbox, mask = parse_box(b, masks, i)
                
                obj_type = 'people' if cls == config.PERSON_CLASS_ID else 'luggage'
                target_list = people if obj_type == 'people' else luggage
                
                if tid is not None:
                    current_ids[obj_type].add(tid)
                    # Update consecutive frame counter
                    self.consecutive[obj_type][tid] = self.consecutive[obj_type].get(tid, 0) + 1
                    
                    # Consecutive frame filtering: only keep objects >= threshold
                    if self.consecutive[obj_type][tid] >= config.MIN_CONSECUTIVE_FRAMES:
                        # Update memory
                        if tid in self.memory[obj_type]:
                            bbox = smooth_bbox(bbox, self.memory[obj_type][tid]['bbox'])
                            # Keep the original first_seen_frame
                            first_seen = self.memory[obj_type][tid].get('first_seen_frame', self.frame_count)
                        else:
                            first_seen = self.frame_count
                        self.memory[obj_type][tid] = {'bbox': bbox, 'conf': conf, 'mask': mask, 
                                                       'cls': cls, 'lost': 0, 'frame': self.frame_count,
                                                       'first_seen_frame': first_seen}
                        target_list.append((conf, cls, tid, bbox, mask))
                else:
                    target_list.append((conf, cls, tid, bbox, mask))
        
        # Clean up expired consecutive frame counters
        for obj_type in ['people', 'luggage']:
            for tid in list(self.consecutive[obj_type].keys()):
                if tid not in current_ids[obj_type]:
                    del self.consecutive[obj_type][tid]
        
        # 3. Add objects from memory (interpolation) with adaptive timeout
        for obj_type in ['people', 'luggage']:
            target_list = people if obj_type == 'people' else luggage
            for tid, mem in list(self.memory[obj_type].items()):
                if tid not in current_ids[obj_type]:
                    mem['lost'] += 1
                    
                    # Calculate how long the object has existed
                    existed_frames = mem['frame'] - mem.get('first_seen_frame', mem['frame'])
                    
                    # Adaptive timeout: short-lived objects get shorter timeout
                    if existed_frames < config.STABLE_THRESHOLD_FRAMES:
                        max_lost = config.SHORT_LIVED_LOST_FRAMES
                    else:
                        max_lost = config.MAX_LOST_FRAMES
                    
                    if mem['lost'] <= max_lost:
                        conf = mem['conf'] * (1.0 - mem['lost'] / (max_lost * 2))
                        target_list.append((conf, mem['cls'], tid, mem['bbox'], mem['mask']))
                    else:
                        del self.memory[obj_type][tid]
        
        # 4. Update Re-ID features (people only) - using keys matching reid.py
        for old_id in self.people_features:
            self.people_features[old_id]['still_tracked'] = False
        
        updated_people = []
        for conf, cls, tid, bbox, mask in people:
            if tid is None or cls != config.PERSON_CLASS_ID:
                updated_people.append((conf, cls, tid, bbox, mask))
                continue
            
            new_feat = extract_person_features(frame, bbox)
            new_center = center_from_bbox(bbox)
            orig_id = self.person_id_mapping.get(tid, tid)
            
            # Re-ID matching
            if orig_id not in self.people_features:
                matched_id, sim = match_person_reentry(new_feat, new_center, self.people_features, self.frame_count)
                if matched_id:
                    self.person_id_mapping[tid] = matched_id
                    orig_id = matched_id
                    print(f"[Re-ID] ID {tid} -> {matched_id} (similarity={sim:.3f})")
            
            # Update features - using keys expected by reid.py
            if orig_id not in self.people_features:
                self.people_features[orig_id] = {}
            self.people_features[orig_id].update({
                'features': new_feat, 
                'last_frame': self.frame_count,
                'last_center': new_center, 
                'still_tracked': True, 
                'is_owner': self.people_features.get(orig_id, {}).get('is_owner', False)
            })
            updated_people.append((conf, cls, tid, bbox, mask))
        
        # Clean up expired features
        max_age = {pid: (config.PERSON_OWNER_HISTORY_FRAMES * 2 if data.get('is_owner') 
                         else config.PERSON_FEATURE_HISTORY_FRAMES * 2)
                   for pid, data in self.people_features.items()}
        expired = [pid for pid, data in self.people_features.items() 
                   if self.frame_count - data['last_frame'] > max_age.get(pid, 0)]
        for pid in expired:
            del self.people_features[pid]
            self.person_id_mapping = {k: v for k, v in self.person_id_mapping.items() if v != pid}
        
        # 5. Prepare people data (for ownership calculation)
        people_data = [(center_from_bbox(bbox)[0], center_from_bbox(bbox)[1], 
                       self.person_id_mapping.get(tid, tid), bbox)
                      for _, _, tid, bbox, _ in updated_people if tid is not None]
        people_by_id = {self.person_id_mapping.get(tid, tid): bbox 
                       for _, _, tid, bbox, _ in updated_people if tid is not None}
        
        return updated_people, luggage, people_data, people_by_id

    def render_frame(self, frame, people, luggage, people_data, people_by_id):
        """Render detection results"""
        # 绘制人物
        for conf, cls, tid, bbox, mask in people:
            is_interp = tid and tid in self.memory['people'] and self.memory['people'][tid]['lost'] > 0
            orig_id = self.person_id_mapping.get(tid, tid)
            reid_tag = f"[ReID:{orig_id}] " if tid in self.person_id_mapping else ""
            label = f"{reid_tag}person id={tid if tid not in self.person_id_mapping else f'{orig_id}(new)'} {conf:.2f}"
            if is_interp:
                label += " [MEM]"
            frame = draw_detection(frame, bbox, mask, label, (255, 0, 0), is_interp)

        # Draw luggage
        for conf, cls, tid, bbox, mask in luggage:
            if tid is None:
                label = f"luggage {conf:.2f}\nNO_ID"
                frame = draw_detection(frame, bbox, mask, label, (0, 0, 255))
                continue
            
            state, owner_text, dist, _ = update_luggage_state(
                self.luggage_ownership_state, tid, bbox, people_data, 
                self.frame_count, self.person_id_mapping)
            
            is_interp = tid in self.memory['luggage'] and self.memory['luggage'][tid]['lost'] > 0
            label = f"luggage id={tid} {conf:.2f}{' [MEM]' if is_interp else ''}\n{owner_text}"
            frame = draw_detection(frame, bbox, mask, label, (0, 0, 255), is_interp)
            
            if state['owner_id'] and state['owner_id'] in people_by_id:
                frame = draw_owner_connection(frame, people_by_id[state['owner_id']], bbox, dist)

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
            elapsed = (self.cap.get(cv2.CAP_PROP_POS_MSEC) - self.start_msec) / 1000.0
            while (time.perf_counter() - self.start_wall) < max(0.0, elapsed):
                time.sleep(0.001)

            # Process and render
            people, luggage, people_data, people_by_id = self.process_frame(frame)
            frame = self.render_frame(frame, people, luggage, people_data, people_by_id)

            if self.frame_count % 30 == 0:
                print(f"[Frame {self.frame_count}] People:{len(people)} Luggage:{len(luggage)}")

            cv2.imshow("view", frame)
            if self.out and self.out.isOpened():
                self.out.write(frame)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

        self.cap.release()
        if self.out:
            self.out.release()
            print(f"\n[Done] Video saved: {self.final_output_path}")
        cv2.destroyAllWindows()


def main():
    """Program entry point"""
    print("=" * 60)
    print("Intelligent Luggage Monitoring System v3.1 (Simplified Version)")
    print("=" * 60)
    
    model = YOLO(config.MODEL_PATH)
    monitor = LuggageMonitor(model)
    monitor.run(config.VIDEO_PATH)


if __name__ == "__main__":
    main()

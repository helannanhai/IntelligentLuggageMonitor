import time
import math
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
video_path = r"Dataset\JiMeng\2.mp4"

PERSON = 0
LUGGAGE = {24, 26, 28}
conf_thres = 0.6

disp_w, disp_h = 1280, 720

def center_xyxy(x1, y1, x2, y2):
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {video_path}")

cv2.namedWindow("out", cv2.WINDOW_NORMAL)
cv2.resizeWindow("out", disp_w, disp_h)

start_wall = time.perf_counter()
start_msec = cap.get(cv2.CAP_PROP_POS_MSEC)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.resize(frame, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    # Time sync to video timestamp
    cur_msec = cap.get(cv2.CAP_PROP_POS_MSEC) - start_msec
    target_sec = max(0.0, cur_msec / 1000.0)
    while (time.perf_counter() - start_wall) < target_sec:
        time.sleep(0.001)

    r = model.predict(frame, conf=conf_thres, classes=[PERSON, 24, 26, 28], verbose=False)[0]

    best_person = None   # (conf, (cx,cy), (x1,y1,x2,y2))
    best_luggage = None

    if r.boxes is not None and len(r.boxes) > 0:
        for b in r.boxes:
            cls_id = int(b.cls.item())
            conf = float(b.conf.item())
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cx, cy = center_xyxy(x1, y1, x2, y2)

            if cls_id == PERSON:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(frame, f"person {conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
                if best_person is None or conf > best_person[0]:
                    best_person = (conf, (cx, cy), (x1, y1, x2, y2))

            elif cls_id in LUGGAGE:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(frame, f"luggage {conf:.2f}", (int(x1), max(0, int(y1) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                if best_luggage is None or conf > best_luggage[0]:
                    best_luggage = (conf, (cx, cy), (x1, y1, x2, y2))

    if best_person and best_luggage:
        pc = best_person[1]
        lc = best_luggage[1]
        dist_px = math.hypot(pc[0] - lc[0], pc[1] - lc[1])

        p_pt = (int(pc[0]), int(pc[1]))
        l_pt = (int(lc[0]), int(lc[1]))
        cv2.line(frame, p_pt, l_pt, (0, 255, 255), 2, cv2.LINE_AA)

        mid = ((p_pt[0] + l_pt[0]) // 2, (p_pt[1] + l_pt[1]) // 2)
        cv2.putText(frame, f"{dist_px:.1f}px", (mid[0] + 8, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("out", frame)

    if (cv2.waitKey(1) & 0xFF) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import time
import csv
import cv2
import psutil
from picamera2 import Picamera2
from ultralytics import YOLO

MODEL_PATH = "yolov8n-pose.pt"
LOG_PATH = "cpu_pose_benchmark.csv"

def temp_c():
    with open("/sys/class/thermal/thermal_zone0/temp") as f:
        return int(f.read()) / 1000

model = YOLO(MODEL_PATH)

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)
picam2.start()
time.sleep(1)

with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "fps", "infer_ms", "cpu_percent", "ram_percent", "temp_c", "people"])

    while True:
        frame = picam2.capture_array()

        t0 = time.time()
        results = model(frame, imgsz=320, conf=0.4, verbose=False)
        infer_ms = (time.time() - t0) * 1000
        fps = 1000 / infer_ms if infer_ms > 0 else 0

        people = len(results[0].keypoints.xy) if results[0].keypoints is not None else 0
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        temp = temp_c()

        annotated = results[0].plot()

        text = f"FPS {fps:.1f} | {infer_ms:.0f} ms | CPU {cpu:.0f}% | RAM {ram:.0f}% | {temp:.1f}C | people {people}"
        cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        writer.writerow([time.time(), fps, infer_ms, cpu, ram, temp, people])
        f.flush()

        cv2.imshow("YOLO Pose CPU Benchmark", annotated)

        print(text)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

picam2.stop()
cv2.destroyAllWindows()
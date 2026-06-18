import time
import cv2
import psutil
from picamera2 import Picamera2
from ultralytics import YOLO


def get_temp_c():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000
    except Exception:
        return None


model = YOLO("yolov8n-pose.pt")

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}
    )
)
picam2.start()
time.sleep(1)

prev_time = time.time()
frame_count = 0

while True:
    frame = picam2.capture_array()

    start = time.time()

    results = model(frame, imgsz=320, conf=0.4, verbose=False)

    annotated = results[0].plot()

    infer_time = time.time() - start
    frame_count += 1

    now = time.time()
    fps = 1 / infer_time if infer_time > 0 else 0

    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory().percent
    temp = get_temp_c()

    text = f"FPS: {fps:.1f} | CPU: {cpu:.0f}% | RAM: {ram:.0f}%"
    if temp is not None:
        text += f" | Temp: {temp:.1f}C"

    cv2.putText(
        annotated,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    cv2.imshow("YOLO Pose CPU Monitor", annotated)

    print(text)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
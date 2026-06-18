"""
main.py — Tecolotl Retail
=========================
Runs shelf-attention analysis using:

- Normal Raspberry Pi camera
- YOLO pose estimation running directly on the Raspberry Pi CPU/GPU
- Existing shelf_attention.py logic

This version replaces the previous IMX500-specific pipeline.

Previous pipeline:
    Camera + IMX500 -> metadata -> HigherHRNet -> Pose -> shelf_attention

New pipeline:
    Normal camera -> frame -> YOLO pose -> Pose -> shelf_attention

The business logic remains the same:
    - Detect people
    - Estimate whether they are facing the shelf
    - Assign them to a shelf zone
    - Display debug overlays
"""

import time
import cv2
import numpy as np
import os
import psutil
from picamera2 import Picamera2

from pose_detector import YOLOPoseDetector, Pose, print_pose
from shelf_attention import (
    analyze_pose,
    get_zone_boundaries,
    ZONE_COUNT,
    IMAGE_WIDTH,
)
from config import SHOW_RULER, SHOW_ZONE_OVERLAY, SHOW_POSE_OVERLAY


# ---------------------------------------------------------------------------
# Camera / model config
# ---------------------------------------------------------------------------

# Normal camera resolution.
# Keep this aligned with shelf_attention.IMAGE_WIDTH.
IMG_W = IMAGE_WIDTH
IMG_H = 480

# YOLO pose model.
# Start with nano because Raspberry Pi has limited compute.
MODEL_PATH = "yolov8n-pose.pt"

# Lower image size = faster, less accurate.
# Higher image size = slower, more accurate.
YOLO_IMAGE_SIZE = 320

# Minimum person confidence.
YOLO_CONFIDENCE = 0.30


# ---------------------------------------------------------------------------
# Display / overlay config
# ---------------------------------------------------------------------------

WINDOW_NAME = "Tecolotl — YOLO Shelf Attention Debug"

# Zone fill colors (B, G, R) — cycles if ZONE_COUNT > len
ZONE_COLORS_BGR = [
    (219, 152,  52),  # azul
    (171, 148,  31),  # verde-azulado
    ( 36, 160, 250),  # amarillo
    (177, 130, 212),  # lila
    (106, 195, 139),  # verde
    ( 78, 186, 244),  # naranja
]

# Status colors (B, G, R)
COLOR_FACING  = ( 86, 199,  29)  # verde — mirando anaquel
COLOR_AWAY    = ( 36, 112, 237)  # azul — de frente, no mira anaquel
COLOR_UNKNOWN = (130, 130, 130)  # gris — inconclusivo

ALPHA_ZONE     = 0.18
LINE_COLOR     = (255, 255, 255)
LINE_THICKNESS = 2

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_THICK = 1


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_ruler(frame: np.ndarray) -> np.ndarray:
    """
    Dibuja una regla de píxeles en la parte inferior del frame.
    Marcas cada 10px, números cada 50px.

    This is useful when calibrating:
    - zone boundaries
    - shoulder width threshold
    - shelf position
    """
    h, w = frame.shape[:2]
    ruler_y = h - 30

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, ruler_y - 4), (w, h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    for x in range(0, w + 1, 10):
        is_major = x % 50 == 0
        color = (0, 255, 255) if is_major else (0, 160, 160)
        tick = 14 if is_major else 7

        cv2.line(frame, (x, ruler_y), (x, ruler_y + tick), color, 1)

        if is_major and x > 0:
            label = str(x)
            lw, _ = cv2.getTextSize(label, FONT, 0.35, 1)[0]
            cv2.putText(
                frame,
                label,
                (x - lw // 2, h - 2),
                FONT,
                0.35,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return frame


def draw_zones(frame: np.ndarray, zone_count: int, img_width: int) -> np.ndarray:
    """
    Dibuja rellenos semi-transparentes por zona y líneas divisoras verticales.

    The actual shelf zone calculation is done in shelf_attention.py.
    This function only draws the visual overlay.
    """
    h, w = frame.shape[:2]
    overlay = frame.copy()

    boundaries = get_zone_boundaries(zone_count, img_width)
    boundaries_px = [int(b * w / img_width) for b in boundaries]
    zone_edges = [0] + boundaries_px + [w]

    for i in range(zone_count):
        x1 = zone_edges[i]
        x2 = zone_edges[i + 1]
        color = ZONE_COLORS_BGR[i % len(ZONE_COLORS_BGR)]

        # Relleno de zona
        cv2.rectangle(overlay, (x1, 0), (x2, h), color, -1)

        # Etiqueta de zona
        label = f"Zona {i + 1}"
        lw, _ = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)[0]
        lx = x1 + (x2 - x1 - lw) // 2

        cv2.putText(
            overlay,
            label,
            (lx, 28),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICK,
            cv2.LINE_AA,
        )

        # Rango en píxeles
        px_start = int(i * img_width / zone_count)
        px_end = int((i + 1) * img_width / zone_count)
        sub = f"{px_start}-{px_end}px"

        sw2, _ = cv2.getTextSize(sub, FONT, 0.38, 1)[0]
        cv2.putText(
            overlay,
            sub,
            (x1 + (x2 - x1 - sw2) // 2, 46),
            FONT,
            0.38,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    # Mezclar relleno con frame original
    frame = cv2.addWeighted(overlay, ALPHA_ZONE, frame, 1 - ALPHA_ZONE, 0)

    # Líneas divisoras nítidas
    for bx in boundaries_px:
        cv2.line(
            frame,
            (bx, 0),
            (bx, h),
            LINE_COLOR,
            LINE_THICKNESS,
            cv2.LINE_AA,
        )

    return frame


def draw_person(
    frame: np.ndarray,
    pose: Pose,
    analysis: dict,
    person_idx: int,
    img_width: int,
    img_height: int,
) -> None:
    """
    Dibuja hombros, línea entre hombros y badge de estado.

    We intentionally do not draw the full bounding box because the current
    shelf-attention logic depends mainly on shoulder keypoints.

    Future idea:
    - Draw wrists and elbows to debug reaching behavior.
    """
    h, w = frame.shape[:2]

    # Scale coordinates in case displayed frame size differs from model frame size.
    sx = w / img_width
    sy = h / img_height

    facing = analysis["facing_shelf"]
    zone = analysis["zone"]

    if facing is True:
        color = COLOR_FACING
        status = "mirando anaquel"
    elif facing is False:
        color = COLOR_AWAY
        status = "de frente"
    else:
        color = COLOR_UNKNOWN
        status = "inconclusivo"

    zone_str = f"Z{zone}" if zone is not None else "Z?"

    ls = pose.left_shoulder
    rs = pose.right_shoulder

    # If shoulders are valid, use them as visual anchor.
    if ls.is_valid() and rs.is_valid():
        lsp = (int(ls.x * sx), int(ls.y * sy))
        rsp = (int(rs.x * sx), int(rs.y * sy))

        # Draw shoulders
        cv2.circle(frame, lsp, 5, color, -1, cv2.LINE_AA)
        cv2.circle(frame, rsp, 5, color, -1, cv2.LINE_AA)

        # Draw shoulder line
        cv2.line(frame, lsp, rsp, color, 2, cv2.LINE_AA)

        # Badge position: above shoulder center
        badge_x = int(((ls.x + rs.x) / 2.0) * sx)
        badge_y = int((min(ls.y, rs.y) - 20) * sy)

    else:
        # If shoulders are not reliable, use any valid keypoint as fallback.
        valid_kps = [kp for kp in pose.keypoints if kp.is_valid()]

        if not valid_kps:
            return

        avg_x = sum(kp.x for kp in valid_kps) / len(valid_kps)
        min_y = min(kp.y for kp in valid_kps)

        badge_x = int(avg_x * sx)
        badge_y = int((min_y - 20) * sy)

    # Badge text
    badge = f"P{person_idx + 1} {zone_str} | {status}"
    (tw, th), baseline = cv2.getTextSize(badge, FONT, FONT_SCALE, FONT_THICK)

    pad = 5

    x1 = max(0, badge_x - tw // 2 - pad)
    y1 = max(0, badge_y - th - baseline - pad * 2)
    x2 = min(w - 1, x1 + tw + pad * 2)
    y2 = min(h - 1, y1 + th + baseline + pad * 2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)

    cv2.putText(
        frame,
        badge,
        (x1 + pad, y2 - baseline - pad),
        FONT,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICK,
        cv2.LINE_AA,
    )

def get_cpu_temp():
    """
    Returns Raspberry Pi CPU temperature in Celsius.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read()) / 1000.0
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

# Load YOLO once.
# This is important. Do not load the model inside the while loop.
pose_detector = YOLOPoseDetector(
    model_path=MODEL_PATH,
    confidence_threshold=YOLO_CONFIDENCE,
    image_size=YOLO_IMAGE_SIZE,
    debug=True,
)

# Start normal Raspberry Pi camera.
# No IMX500 object is needed anymore.
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (IMG_W, IMG_H), "format": "RGB888"},
    buffer_count=4,
)

picam2.configure(config)
picam2.start()

print(f"Cámara iniciada. Zonas: {ZONE_COUNT} | Límites: {get_zone_boundaries()}")
print(f"Modelo YOLO: {MODEL_PATH} | imgsz={YOLO_IMAGE_SIZE} | conf={YOLO_CONFIDENCE}")
print("Presiona 'q' para salir.\n")

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

last_debug_time = 0
last_frame_time = time.monotonic()

try:
    while True:
        # -------------------------------------------------------------------
        # Capture frame
        # -------------------------------------------------------------------
        # picamera2 returns RGB.
        # OpenCV and YOLO through OpenCV commonly work with BGR.
        frame_rgb = picam2.capture_array("main")
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # -------------------------------------------------------------------
        # Optional overlays
        # -------------------------------------------------------------------
        if SHOW_ZONE_OVERLAY:
            frame = draw_zones(frame, ZONE_COUNT, IMAGE_WIDTH)

        if SHOW_RULER:
            frame = draw_ruler(frame)

        # -------------------------------------------------------------------
        # Pose detection with YOLO
        # -------------------------------------------------------------------
        poses = pose_detector.get_poses(frame)

        # -------------------------------------------------------------------
        # FPS/debug info
        # -------------------------------------------------------------------
        now = time.monotonic()
        dt = now - last_frame_time
        last_frame_time = now

        fps = 1.0 / dt if dt > 0 else 0.0

        if now - last_debug_time > 5:
            last_debug_time = now
            print(f"[DEBUG] poses: {len(poses)} | approx FPS: {fps:.1f}")

        # -------------------------------------------------------------------
        # Analyze each detected person
        # -------------------------------------------------------------------
        if poses:
            for i, pose in enumerate(poses):
                analysis = analyze_pose(pose)

                if SHOW_POSE_OVERLAY:
                    draw_person(frame, pose, analysis, i, IMAGE_WIDTH, IMG_H)

                # Console output
                facing = analysis["facing_shelf"]
                zone = analysis["zone"]

                if facing is True:
                    orient = "DE ESPALDAS — mirando anaquel"
                elif facing is False:
                    orient = "DE FRENTE — no mira anaquel"
                else:
                    orient = "DE LADO — inconclusivo"

                print(f"Persona {i + 1}: {orient} | Zona {zone}")

        else:
            cv2.putText(
                frame,
                "Sin detecciones...",
                (12, IMG_H - 12),
                FONT,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        # -------------------------------------------------------------------
        # System monitor overlay
        # -------------------------------------------------------------------
        temp = get_cpu_temp()
        cpu_usage = psutil.cpu_percent()
        cv2.putText(
            frame,
            f"CPU: {temp:.1f} C | Load: {cpu_usage:.0f}% | FPS: {fps:.1f}",
            (10, 25),
            FONT,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        # -------------------------------------------------------------------
        # Display frame
        # -------------------------------------------------------------------
        cv2.imshow(WINDOW_NAME, frame)

        # Press q to exit.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Small sleep to reduce CPU pressure.
        # Remove or lower this if you want maximum FPS.
        time.sleep(0.02)

except KeyboardInterrupt:
    print("\nDetenido.")

finally:
    cv2.destroyAllWindows()
    picam2.stop()
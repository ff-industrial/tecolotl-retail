"""
pose_detector.py — Tecolotl Retail
===================================
Runs YOLO pose estimation directly on the Raspberry Pi using a normal camera.

This replaces the previous IMX500/HigherHRNet pipeline.

Previous pipeline:
    IMX500 metadata -> HigherHRNet postprocess -> Pose objects

New pipeline:
    Camera frame -> YOLO pose model -> Pose objects

The important architectural idea is that the rest of the project does not need
to know which model produced the pose. This module still exposes the same
internal data structures:

- Keypoint
- Pose
- print_pose()

That means shelf_attention.py can continue importing:

    from pose_detector import Pose

YOLO pose models use the COCO 17-keypoint format, which matches the structure
we were already using with HigherHRNet.

Recommended starting model for Raspberry Pi:
    yolov8n-pose.pt

For better performance, reduce imgsz to 320.
For better accuracy, increase imgsz to 480 or 640, but FPS will drop.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# COCO keypoint index constants
# ---------------------------------------------------------------------------
# YOLO pose uses the same 17-keypoint COCO order:
#
# 0  nose
# 1  left_eye
# 2  right_eye
# 3  left_ear
# 4  right_ear
# 5  left_shoulder
# 6  right_shoulder
# 7  left_elbow
# 8  right_elbow
# 9  left_wrist
# 10 right_wrist
# 11 left_hip
# 12 right_hip
# 13 left_knee
# 14 right_knee
# 15 left_ankle
# 16 right_ankle
# ---------------------------------------------------------------------------

KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16


# All 17 COCO keypoints — exposed for current and future use.
# Immediately useful:
# - nose
# - shoulders
# - hips
#
# Future retail use:
# - wrists + elbows for detecting if someone reaches toward a shelf
# - eyes/ears for estimating head direction
RETAIL_KEYPOINTS = {
    "nose":            KP_NOSE,
    "left_eye":        KP_LEFT_EYE,
    "right_eye":       KP_RIGHT_EYE,
    "left_ear":        KP_LEFT_EAR,
    "right_ear":       KP_RIGHT_EAR,
    "left_shoulder":   KP_LEFT_SHOULDER,
    "right_shoulder":  KP_RIGHT_SHOULDER,
    "left_elbow":      KP_LEFT_ELBOW,
    "right_elbow":     KP_RIGHT_ELBOW,
    "left_wrist":      KP_LEFT_WRIST,
    "right_wrist":     KP_RIGHT_WRIST,
    "left_hip":        KP_LEFT_HIP,
    "right_hip":       KP_RIGHT_HIP,
    "left_knee":       KP_LEFT_KNEE,
    "right_knee":      KP_RIGHT_KNEE,
    "left_ankle":      KP_LEFT_ANKLE,
    "right_ankle":     KP_RIGHT_ANKLE,
}


# ---------------------------------------------------------------------------
# Default YOLO configuration
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = "yolov8n-pose.pt"
DEFAULT_CONFIDENCE = 0.30
DEFAULT_IMAGE_SIZE = 320


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Keypoint:
    """
    A single body keypoint.

    x:          Horizontal coordinate in image pixels.
    y:          Vertical coordinate in image pixels.
    confidence: Model confidence for this specific keypoint.
    """
    x: float
    y: float
    confidence: float

    def is_valid(self, min_confidence: float = 0.3) -> bool:
        """Returns True if this keypoint has enough confidence to be used."""
        return self.confidence >= min_confidence


@dataclass
class Pose:
    """
    A single detected person with 17 COCO keypoints, a bounding box, and a score.

    keypoints: list of 17 Keypoint objects in COCO order.
    box:       [x1, y1, x2, y2] in image coordinates.
    score:     overall person detection confidence from YOLO.
    """
    keypoints: list[Keypoint]
    box: np.ndarray
    score: float

    def get(self, index: int) -> Keypoint:
        """Return keypoint by COCO index."""
        return self.keypoints[index]

    def retail_keypoints(self) -> dict[str, Keypoint]:
        """Return all named keypoints useful for retail analytics."""
        return {name: self.keypoints[idx] for name, idx in RETAIL_KEYPOINTS.items()}

    @property
    def nose(self) -> Keypoint:
        return self.keypoints[KP_NOSE]

    @property
    def left_shoulder(self) -> Keypoint:
        return self.keypoints[KP_LEFT_SHOULDER]

    @property
    def right_shoulder(self) -> Keypoint:
        return self.keypoints[KP_RIGHT_SHOULDER]

    @property
    def left_elbow(self) -> Keypoint:
        return self.keypoints[KP_LEFT_ELBOW]

    @property
    def right_elbow(self) -> Keypoint:
        return self.keypoints[KP_RIGHT_ELBOW]

    @property
    def left_wrist(self) -> Keypoint:
        return self.keypoints[KP_LEFT_WRIST]

    @property
    def right_wrist(self) -> Keypoint:
        return self.keypoints[KP_RIGHT_WRIST]

    @property
    def left_hip(self) -> Keypoint:
        return self.keypoints[KP_LEFT_HIP]

    @property
    def right_hip(self) -> Keypoint:
        return self.keypoints[KP_RIGHT_HIP]


# ---------------------------------------------------------------------------
# YOLO pose detector
# ---------------------------------------------------------------------------

class YOLOPoseDetector:
    """
    Wrapper around a YOLO pose model.

    This class converts raw Ultralytics YOLO outputs into the internal Pose
    format used by shelf_attention.py.

    Why use a class instead of a simple function?
    ------------------------------------------------
    Because the YOLO model should be loaded only once.

    Bad:
        load model -> detect
        load model -> detect
        load model -> detect

    Good:
        load model once
        detect many frames

    Usage:
        detector = YOLOPoseDetector("yolov8n-pose.pt")
        poses = detector.get_poses(frame_bgr)
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        image_size: int = DEFAULT_IMAGE_SIZE,
        debug: bool = False,
    ):
        """
        Load YOLO pose model.

        Args:
            model_path:            Path/name of YOLO pose model.
            confidence_threshold:  Minimum person detection confidence.
            image_size:            Inference image size. Lower = faster, less accurate.
            debug:                 Prints extra information during startup.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.debug = debug

        if self.debug:
            print(f"[pose_detector] Loading YOLO model: {self.model_path}")

        self.model = YOLO(self.model_path)

        if self.debug:
            print("[pose_detector] YOLO model loaded successfully")

    def get_poses(self, frame_bgr: np.ndarray) -> list[Pose]:
        """
        Run YOLO pose estimation on a BGR OpenCV frame.

        Args:
            frame_bgr:
                Image frame in BGR format.
                This is the normal format used by OpenCV.

        Returns:
            List of Pose objects.
            Empty list if no people are detected.
        """

        # Run YOLO inference.
        # verbose=False keeps the terminal clean.
        results = self.model(
            frame_bgr,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            verbose=False,
        )

        if not results:
            return []

        result = results[0]

        # If YOLO found no pose/keypoint data, return no detections.
        if result.keypoints is None or result.boxes is None:
            return []

        # YOLO keypoints:
        # xy shape:   (N, 17, 2)
        # conf shape: (N, 17)
        #
        # N = number of detected people.
        keypoints_xy = result.keypoints.xy.cpu().numpy()
        keypoints_conf = result.keypoints.conf.cpu().numpy()

        # YOLO boxes:
        # boxes_xyxy shape: (N, 4)
        # scores shape:    (N,)
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()

        poses: list[Pose] = []

        for i in range(len(keypoints_xy)):
            kps: list[Keypoint] = []

            for j in range(17):
                x = float(keypoints_xy[i][j][0])
                y = float(keypoints_xy[i][j][1])
                conf = float(keypoints_conf[i][j])

                kps.append(Keypoint(x=x, y=y, confidence=conf))

            poses.append(
                Pose(
                    keypoints=kps,
                    box=np.array(boxes_xyxy[i]),
                    score=float(scores[i]),
                )
            )

        return poses


# ---------------------------------------------------------------------------
# Compatibility helper
# ---------------------------------------------------------------------------

def get_poses(
    frame_bgr: np.ndarray,
    detector: YOLOPoseDetector,
) -> list[Pose]:
    """
    Small compatibility function.

    This keeps a similar name to the previous IMX500 version, but now receives:

        frame_bgr + detector

    instead of:

        metadata + imx500

    Recommended usage in main:
        detector = YOLOPoseDetector()
        poses = detector.get_poses(frame)

    Alternative usage:
        poses = get_poses(frame, detector)
    """
    return detector.get_poses(frame_bgr)


# ---------------------------------------------------------------------------
# Debug utility — print keypoints for a single pose
# ---------------------------------------------------------------------------

def print_pose(pose: Pose) -> None:
    """Print a human-readable summary of a Pose. Useful during development."""
    print(f"  Score: {pose.score:.2f}  Box: {pose.box}")

    for name, kp in pose.retail_keypoints().items():
        status = "✓" if kp.is_valid() else "✗"
        print(
            f"  [{status}] {name:20s}  "
            f"x={kp.x:.1f}  y={kp.y:.1f}  conf={kp.confidence:.2f}"
        )
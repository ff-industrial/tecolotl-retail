import cv2
import mediapipe as mp

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo leer la cámara")
        break

    # OpenCV usa BGR, MediaPipe usa RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar pose
    results = pose.process(rgb)

    # Dibujar keypoints
    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Ejemplo: obtener nariz
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        h, w, _ = frame.shape
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)

        cv2.circle(frame, (nose_x, nose_y), 8, (0, 255, 0), -1)

    cv2.imshow("Pose Detection - Raspberry Pi", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
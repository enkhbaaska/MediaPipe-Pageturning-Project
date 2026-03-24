import cv2
import mediapipe as mp
import time
from pynput.keyboard import Controller, Key
import math

keyboard = Controller()

# ---------- TUNABLE SETTINGS ----------
CAMERA_INDEX = 1  # try 0/1 depending on your webcam

# Mouth open ratio thresholds (hysteresis)
# - opens when ratio > OPEN_TH
# - closes when ratio < CLOSE_TH
MOUTH_OPEN_TH = 0.35
MOUTH_CLOSE_TH = 0.28  # must be < MOUTH_OPEN_TH

# Require mouth-open for a few frames before triggering
OPEN_FRAMES_TO_TRIGGER = 3

# After a trigger, ignore inputs for this many seconds
COOLDOWN = 0.9

SHOW_OVERLAY = True
# -------------------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# State
last_trigger = 0.0
mouth_is_open = False
mouth_open_frames = 0

# FaceMesh mouth landmarks:
# corners: 61 (left), 291 (right)
# inner lips: 13 (upper), 14 (lower)
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_UPPER = 13
MOUTH_LOWER = 14

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def press_next():
    keyboard.press(Key.right)
    keyboard.release(Key.right)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    now = time.time()

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark

        def to_xy(i):
            return (lm[i].x * w, lm[i].y * h)

        left_corner = to_xy(MOUTH_LEFT)
        right_corner = to_xy(MOUTH_RIGHT)
        upper_lip = to_xy(MOUTH_UPPER)
        lower_lip = to_xy(MOUTH_LOWER)

        mouth_width = dist(left_corner, right_corner) + 1e-6
        mouth_gap = dist(upper_lip, lower_lip)

        # Ratio is scale-invariant: works at different distances from camera
        mouth_ratio = mouth_gap / mouth_width

        # Hysteresis to avoid flicker
        if mouth_is_open:
            mouth_is_open = mouth_ratio > MOUTH_CLOSE_TH
        else:
            mouth_is_open = mouth_ratio > MOUTH_OPEN_TH

        # Count consecutive frames mouth is open
        if mouth_is_open:
            mouth_open_frames += 1
        else:
            mouth_open_frames = 0

        can_trigger = (now - last_trigger) > COOLDOWN

        if can_trigger and mouth_open_frames >= OPEN_FRAMES_TO_TRIGGER:
            print("Mouth open → Next page")
            press_next()
            last_trigger = now
            mouth_open_frames = 0  # reset so it doesn't spam

        if SHOW_OVERLAY:
            txt = (
                f"mouth_ratio={mouth_ratio:.3f}  "
                f"open>{MOUTH_OPEN_TH:.2f} close<{MOUTH_CLOSE_TH:.2f}  "
                f"frames={mouth_open_frames}"
            )
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "Open mouth = Next page | ESC=Quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Mouth Page Turn", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

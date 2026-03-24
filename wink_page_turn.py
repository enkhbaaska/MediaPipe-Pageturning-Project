import cv2
import mediapipe as mp
import time
from pynput.keyboard import Controller, Key
import math

keyboard = Controller()

# ---------- TUNABLE SETTINGS ----------
CAMERA_INDEX = 1

# Eye Aspect Ratio threshold: lower = needs more closure
EAR_CLOSED_THRESHOLD = 0.19

# Wink stability
WINK_FRAMES = 2          # consecutive frames one eye must be closed
COOLDOWN = 0.9           # seconds after a trigger to ignore inputs

# Both-eyes blink suppression (prevents accidental page turns)
BOTH_BLINK_MIN_TIME = 0.10   # both eyes must be closed this long to count as a real blink
BOTH_BLINK_SUPPRESS = 0.25   # seconds to ignore winks after a real blink

# Extra guard: block winks near any both-eye closure (fixes double-blink -> page turn)
BLINK_GUARD = 0.25           # seconds to block winks around both-eye blink edges

SHOW_OVERLAY = True
# -------------------------------------

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

# Camera (CAP_DSHOW often helps on Windows)
cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# State
last_trigger = 0.0
left_closed_frames = 0
right_closed_frames = 0

both_closed_since = None
both_closed_until = 0.0
last_both_closed_time = -1e9

# MediaPipe FaceMesh eye indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def ear(pts):
    # (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)

def press_next():
    keyboard.press(Key.right)
    keyboard.release(Key.right)

def press_prev():
    keyboard.press(Key.left)
    keyboard.release(Key.left)

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

        left_pts = [to_xy(i) for i in LEFT_EYE]
        right_pts = [to_xy(i) for i in RIGHT_EYE]

        left_ear = ear(left_pts)
        right_ear = ear(right_pts)

        left_closed = left_ear < EAR_CLOSED_THRESHOLD
        right_closed = right_ear < EAR_CLOSED_THRESHOLD

        # Update consecutive-closed frame counters
        left_closed_frames = left_closed_frames + 1 if left_closed else 0
        right_closed_frames = right_closed_frames + 1 if right_closed else 0

        # Detect both-eyes closure and remember when it happens (for guarding)
        if left_closed and right_closed:
            last_both_closed_time = now
            both_closed_since = both_closed_since or now
        else:
            # if we just ended a both-closed period, decide whether to suppress
            if both_closed_since and (now - both_closed_since) >= BOTH_BLINK_MIN_TIME:
                both_closed_until = now + BOTH_BLINK_SUPPRESS
            both_closed_since = None

        can_trigger = (
            (now - last_trigger) > COOLDOWN
            and now >= both_closed_until
            and (now - last_both_closed_time) > BLINK_GUARD
        )

        if can_trigger:
            # Left wink -> previous page
            if left_closed_frames >= WINK_FRAMES and not right_closed:
                print("Left wink → Previous page")
                press_prev()
                last_trigger = now
                left_closed_frames = 0

            # Right wink -> next page
            elif right_closed_frames >= WINK_FRAMES and not left_closed:
                print("Right wink → Next page")
                press_next()
                last_trigger = now
                right_closed_frames = 0

        if SHOW_OVERLAY:
            txt = (
                f"L_EAR={left_ear:.3f} R_EAR={right_ear:.3f} thr={EAR_CLOSED_THRESHOLD:.2f} "
                f"guard={max(0.0, BLINK_GUARD - (now - last_both_closed_time)):.2f}s"
            )
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, "Wink: L=Prev, R=Next | ESC=Quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Wink Page Turn (SumatraPDF)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

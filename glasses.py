import cv2
import mediapipe as mp
import time
from pynput.keyboard import Controller, Key
import math

keyboard = Controller()

# ---------- TUNABLE SETTINGS ----------
CAMERA_INDEX = 1

# Hysteresis thresholds (good for glasses jitter)
EAR_CLOSE_TH = 0.18
EAR_OPEN_TH  = 0.22   # must be > EAR_CLOSE_TH

# Smoothing for glasses jitter (lower = smoother but more latency)
EAR_EMA_ALPHA = 0.25  # try 0.20–0.30

# Wink stability
WINK_FRAMES = 2
COOLDOWN = 0.9

# Both-eyes blink suppression
BOTH_BLINK_MIN_TIME = 0.10
BOTH_BLINK_SUPPRESS = 0.25

# If one eye closes and the other closes within this time,
# treat as a normal blink (NOT a wink).
BLINK_PAIR_WINDOW = 0.15  # 120ms (increase to 0.15 if needed)

SHOW_OVERLAY = True
# -------------------------------------

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

# State
last_trigger = 0.0
left_closed_frames = 0
right_closed_frames = 0

both_closed_since = None
both_closed_until = 0.0

# Smoothed EAR + hysteresis eye state
left_ear_ema = None
right_ear_ema = None
left_is_closed = False
right_is_closed = False

# Blink-pairing state
first_eye_close_time = None   # when first eye in a blink started closing
paired_blink_until = 0.0      # block winks until this time after paired blink

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def ear(pts):
    p1, p2, p3, p4, p5, p6 = pts
    return (dist(p2, p6) + dist(p3, p5)) / (2.0 * dist(p1, p4) + 1e-6)

def ema(prev, x, alpha):
    return x if prev is None else (alpha * x + (1 - alpha) * prev)

def update_closed(is_closed, ear_value):
    if is_closed:
        return False if ear_value > EAR_OPEN_TH else True
    else:
        return True if ear_value < EAR_CLOSE_TH else False

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

        left_ear_raw = ear(left_pts)
        right_ear_raw = ear(right_pts)

        left_ear_ema = ema(left_ear_ema, left_ear_raw, EAR_EMA_ALPHA)
        right_ear_ema = ema(right_ear_ema, right_ear_raw, EAR_EMA_ALPHA)

        prev_left_closed = left_is_closed
        prev_right_closed = right_is_closed

        left_is_closed = update_closed(left_is_closed, left_ear_ema)
        right_is_closed = update_closed(right_is_closed, right_ear_ema)

        # Count closed frames
        left_closed_frames = left_closed_frames + 1 if left_is_closed else 0
        right_closed_frames = right_closed_frames + 1 if right_is_closed else 0

        # ---- Blink pairing window (fixes "double blink -> left wink") ----
        # detect transitions open->closed
        left_just_closed = (not prev_left_closed) and left_is_closed
        right_just_closed = (not prev_right_closed) and right_is_closed

        # if either eye just closed, start a "maybe blink" timer
        if left_just_closed or right_just_closed:
            if first_eye_close_time is None:
                first_eye_close_time = now

        # if the other eye closes within the pairing window, suppress winks
        if first_eye_close_time is not None:
            if (left_is_closed and right_is_closed) or (left_just_closed and right_is_closed) or (right_just_closed and left_is_closed):
                if (now - first_eye_close_time) <= BLINK_PAIR_WINDOW:
                    paired_blink_until = now + BOTH_BLINK_SUPPRESS  # block winks briefly
            # reset if window expired or both eyes reopened
            if (now - first_eye_close_time) > BLINK_PAIR_WINDOW or (not left_is_closed and not right_is_closed):
                first_eye_close_time = None
        # ---------------------------------------------------------------

        # Classic both-closed timing suppress (still useful)
        if left_is_closed and right_is_closed:
            both_closed_since = both_closed_since or now
        else:
            if both_closed_since and (now - both_closed_since) >= BOTH_BLINK_MIN_TIME:
                both_closed_until = now + BOTH_BLINK_SUPPRESS
            both_closed_since = None

        can_trigger = (
            (now - last_trigger) > COOLDOWN
            and now >= both_closed_until
            and now >= paired_blink_until
        )

        if can_trigger:
            # Left wink
            if left_closed_frames >= WINK_FRAMES and not right_is_closed:
                print("Left wink → Previous page")
                press_prev()
                last_trigger = now
                left_closed_frames = 0

            # Right wink
            elif right_closed_frames >= WINK_FRAMES and not left_is_closed:
                print("Right wink → Next page")
                press_next()
                last_trigger = now
                right_closed_frames = 0

        if SHOW_OVERLAY:
            txt = (
                f"L={left_ear_ema:.3f} R={right_ear_ema:.3f} "
                f"close<{EAR_CLOSE_TH:.2f} open>{EAR_OPEN_TH:.2f} "
                f"pair_guard={max(0.0, paired_blink_until-now):.2f}s"
            )
            cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, "Wink: L=Prev, R=Next | ESC=Quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Wink Page Turn", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

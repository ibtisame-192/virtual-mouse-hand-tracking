import cv2
import mediapipe as mp
import pyautogui
import math
import time
import numpy as np
from collections import deque

# ── Config ───────────────────────────────────────────────────────────────────
CLICK_DISTANCE      = 0.04   # normalised pinch threshold
CLICK_HOLD_FRAMES   = 3      # frames pinch must be held before registering click
RELEASE_HOLD_FRAMES = 2      # frames pinch must be open before re-arming click
SCREEN_MARGIN       = 0.10   # edge buffer fraction
MODEL_PATH          = "hand_landmarker.task"
CAMERA_INDEX        = 0

# Kalman tuning
PROCESS_NOISE       = 1e-2   # lower = trust model more (smoother, more lag)
MEASURE_NOISE       = 1e-1   # lower = trust measurement more (snappier, more jitter)

# Multi-frame averaging window
LANDMARK_HISTORY    = 5      # frames to average landmarks over
# ─────────────────────────────────────────────────────────────────────────────

pyautogui.FAILSAFE  = False
pyautogui.PAUSE     = 0

screen_w, screen_h  = pyautogui.size()

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode


# ── Kalman Filter (2D position + velocity) ───────────────────────────────────
class KalmanMouse:
    """
    4-state Kalman filter: [x, y, vx, vy]
    Predicts position from velocity, corrects with measured screen coords.
    """
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)

        # Transition: x' = x + vx, y' = y + vy
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement picks out just x, y
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * PROCESS_NOISE
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * MEASURE_NOISE
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)

        self.initialised = False

    def update(self, x: float, y: float) -> tuple[int, int]:
        measurement = np.array([[x], [y]], dtype=np.float32)
        if not self.initialised:
            self.kf.statePost = np.array(
                [[x], [y], [0], [0]], dtype=np.float32)
            self.initialised = True

        self.kf.predict()
        corrected = self.kf.correct(measurement)
        return int(corrected[0]), int(corrected[1])


# ── Landmark averager ────────────────────────────────────────────────────────
class LandmarkAverager:
    """
    Keeps a rolling window of landmark positions and returns their mean,
    removing high-frequency jitter before it ever reaches the Kalman filter.
    """
    def __init__(self, num_landmarks: int = 21, history: int = LANDMARK_HISTORY):
        # deque per landmark, each entry is (x, y, z)
        self._buf: list[deque] = [
            deque(maxlen=history) for _ in range(num_landmarks)
        ]

    def update(self, hand) -> list:
        for i, lm in enumerate(hand):
            self._buf[i].append((lm.x, lm.y, lm.z))
        return self._averaged()

    def _averaged(self) -> list:
        class AvgLM:
            __slots__ = ("x", "y", "z")
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        result = []
        for buf in self._buf:
            xs, ys, zs = zip(*buf)
            result.append(AvgLM(
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                sum(zs) / len(zs),
            ))
        return result


# ── Click debouncer ──────────────────────────────────────────────────────────
class ClickDebouncer:
    """
    Requires the pinch to be held for CLICK_HOLD_FRAMES consecutive frames
    before firing, and released for RELEASE_HOLD_FRAMES frames before
    re-arming. Eliminates single-frame false positives entirely.
    """
    def __init__(self):
        self._pinch_count   = 0
        self._release_count = 0
        self._armed         = True
        self.is_clicking    = False   # True while pinch is active + confirmed

    def update(self, pinching: bool) -> bool:
        """Returns True on the frame a real click should fire."""
        fired = False
        if pinching:
            self._release_count = 0
            if self._armed:
                self._pinch_count += 1
                if self._pinch_count >= CLICK_HOLD_FRAMES:
                    if not self.is_clicking:
                        fired = True          # fire exactly once
                    self.is_clicking = True
                    self._armed      = False
        else:
            self._pinch_count = 0
            self.is_clicking  = False
            if not self._armed:
                self._release_count += 1
                if self._release_count >= RELEASE_HOLD_FRAMES:
                    self._armed = True
        return fired


# ── Helpers ──────────────────────────────────────────────────────────────────
def pinch_distance(hand) -> float:
    t, i = hand[4], hand[8]
    return math.hypot(t.x - i.x, t.y - i.y)


def map_to_screen(nx: float, ny: float) -> tuple[float, float]:
    m  = SCREEN_MARGIN
    sx = (nx - m) / (1 - 2 * m)
    sy = (ny - m) / (1 - 2 * m)
    return (
        max(0.0, min(1.0, sx)) * screen_w,
        max(0.0, min(1.0, sy)) * screen_h,
    )


def draw_overlay(frame, hand, distance: float, debouncer: ClickDebouncer, fps: float):
    h, w  = frame.shape[:2]
    pinch = distance < CLICK_DISTANCE

    for lm in hand:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 3, (180, 180, 180), -1)

    for idx, color in {4: (255, 120, 0), 8: (0, 210, 255)}.items():
        px, py = int(hand[idx].x * w), int(hand[idx].y * h)
        cv2.circle(frame, (px, py), 11, color, -1)
        cv2.circle(frame, (px, py), 11, (255, 255, 255), 2)

    t_pt  = (int(hand[4].x * w), int(hand[4].y * h))
    i_pt  = (int(hand[8].x * w), int(hand[8].y * h))
    lcolor = (0, 60, 255) if pinch else (0, 220, 80)
    cv2.line(frame, t_pt, i_pt, lcolor, 2)

    # Pinch progress bar
    bar_max  = 130
    fill     = int(min(distance / (CLICK_DISTANCE * 2), 1.0) * bar_max)
    bcolor   = (0, 60, 255) if pinch else (0, 220, 80)
    cv2.rectangle(frame, (14, 14), (14 + bar_max, 28), (40, 40, 40), -1)
    cv2.rectangle(frame, (14, 14), (14 + fill,    28), bcolor,       -1)
    cv2.putText(frame, "pinch", (14, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

    # Click flash
    if debouncer.is_clicking:
        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 60, 255), -1)
        cv2.addWeighted(ov, 0.13, frame, 0.87, 0, frame)
        cv2.putText(frame, "CLICK", (w // 2 - 44, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 80, 255), 3, cv2.LINE_AA)

    # Pinch hold progress arc
    if pinch and not debouncer.is_clicking:
        progress = debouncer._pinch_count / CLICK_HOLD_FRAMES
        axes     = (22, 22)
        cx, cy   = w - 40, 40
        cv2.ellipse(frame, (cx, cy), axes, -90,
                    0, int(360 * progress), (0, 200, 255), 3)

    cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.putText(frame, "ESC / Q  quit", (14, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (110, 110, 110), 1, cv2.LINE_AA)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )

    kalman    = KalmanMouse()
    averager  = LandmarkAverager()
    debouncer = ClickDebouncer()

    prev_time = time.time()
    fps       = 0.0

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

        # Suggest higher resolution for better landmark accuracy
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            now       = time.time()
            fps       = 0.9 * fps + 0.1 / max(now - prev_time, 1e-6)
            prev_time = now

            if result.hand_landmarks:
                raw_hand = result.hand_landmarks[0]

                # 1️⃣  Multi-frame average → removes jitter
                smooth_hand = averager.update(raw_hand)

                # 2️⃣  Map averaged index-tip to screen coords
                sx, sy = map_to_screen(smooth_hand[8].x, smooth_hand[8].y)

                # 3️⃣  Kalman filter → removes residual noise, predicts ahead
                kx, ky = kalman.update(sx, sy)
                pyautogui.moveTo(kx, ky)

                # 4️⃣  Click with multi-frame debounce
                dist    = pinch_distance(smooth_hand)
                pinching = dist < CLICK_DISTANCE
                if debouncer.update(pinching):
                    pyautogui.click()

                draw_overlay(frame, smooth_hand, dist, debouncer, fps)
            else:
                kalman.initialised = False          # reset on hand loss
                cv2.putText(frame, "No hand detected", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (80, 80, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, f"FPS {fps:.0f}", (frame.shape[1] - 90, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (150, 150, 150), 1, cv2.LINE_AA)

            cv2.imshow("Hand Mouse v2 — Kalman + Averaging + Debounce", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord('q')):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

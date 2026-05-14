import cv2
import mediapipe as mp
import pyautogui
import math
import time

# ── Config ──────────────────────────────────────────────────────────────────
CLICK_DISTANCE      = 0.04   # normalised distance threshold for click
SMOOTHING           = 0.25   # 0 = instant, 1 = never moves (lower = snappier)
SCREEN_MARGIN       = 0.10   # fraction of frame to treat as edge buffer
MODEL_PATH          = "hand_landmarker.task"
CAMERA_INDEX        = 0
# ────────────────────────────────────────────────────────────────────────────

pyautogui.FAILSAFE  = False
pyautogui.PAUSE     = 0

screen_w, screen_h = pyautogui.size()

BaseOptions         = mp.tasks.BaseOptions
HandLandmarker      = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode   = mp.tasks.vision.RunningMode


def pinch_distance(hand) -> float:
    """Euclidean distance (normalised) between thumb tip and index tip."""
    t, i = hand[4], hand[8]
    return math.hypot(t.x - i.x, t.y - i.y)


def map_to_screen(nx: float, ny: float) -> tuple[int, int]:
    """
    Map normalised [0,1] coords to screen pixels, with a margin so the
    cursor can reach the screen edges without moving the hand to the
    very edge of the camera frame.
    """
    m = SCREEN_MARGIN
    sx = (nx - m) / (1 - 2 * m)
    sy = (ny - m) / (1 - 2 * m)
    sx = max(0.0, min(1.0, sx))
    sy = max(0.0, min(1.0, sy))
    return int(sx * screen_w), int(sy * screen_h)


def draw_overlay(frame, hand, distance: float, clicking: bool, fps: float):
    """Draw landmarks, connection lines, distance bar, and status text."""
    h, w = frame.shape[:2]

    # Key landmark indices to highlight
    key_ids = {4: (255, 100, 0), 8: (0, 200, 255)}

    # Draw all landmarks
    for lm in hand:
        px, py = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (px, py), 4, (200, 200, 200), -1)

    # Highlight thumb (4) and index (8)
    for idx, color in key_ids.items():
        px, py = int(hand[idx].x * w), int(hand[idx].y * h)
        cv2.circle(frame, (px, py), 10, color, -1)
        cv2.circle(frame, (px, py), 10, (255, 255, 255), 2)

    # Line between thumb and index
    t  = (int(hand[4].x * w), int(hand[4].y * h))
    ix = (int(hand[8].x * w), int(hand[8].y * h))
    line_color = (0, 80, 255) if clicking else (0, 230, 80)
    cv2.line(frame, t, ix, line_color, 2)

    # Distance bar (top-left)
    bar_max   = 120
    bar_fill  = int(min(distance / (CLICK_DISTANCE * 2), 1.0) * bar_max)
    bar_color = (0, 80, 255) if clicking else (0, 230, 80)
    cv2.rectangle(frame, (16, 16), (16 + bar_max, 30), (50, 50, 50), -1)
    cv2.rectangle(frame, (16, 16), (16 + bar_fill, 30), bar_color, -1)
    cv2.putText(frame, "pinch", (16, 46), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Click flash
    if clicking:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 80, 255), -1)
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.putText(frame, "CLICK", (w // 2 - 40, h // 2),
                    cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 80, 255), 3, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"FPS {fps:.0f}", (w - 90, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)

    # Instructions
    cv2.putText(frame, "ESC / Q  quit", (16, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 130, 130), 1, cv2.LINE_AA)


def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )

    clicked      = False
    smooth_x     = screen_w / 2.0
    smooth_y     = screen_h / 2.0
    prev_time    = time.time()
    fps          = 0.0

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {CAMERA_INDEX}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame     = cv2.flip(frame, 1)
            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Timestamp in milliseconds (required by Tasks API)
            timestamp_ms = int(time.time() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            # FPS
            now      = time.time()
            fps      = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            if result.hand_landmarks:
                hand     = result.hand_landmarks[0]
                distance = pinch_distance(hand)

                # ── Mouse movement with smoothing ──
                target_x, target_y = map_to_screen(hand[8].x, hand[8].y)
                smooth_x = smooth_x + SMOOTHING * (target_x - smooth_x)
                smooth_y = smooth_y + SMOOTHING * (target_y - smooth_y)
                pyautogui.moveTo(int(smooth_x), int(smooth_y))

                # ── Click with debounce ──
                if distance < CLICK_DISTANCE and not clicked:
                    pyautogui.click()
                    clicked = True
                elif distance >= CLICK_DISTANCE:
                    clicked = False

                draw_overlay(frame, hand, distance, clicked, fps)
            else:
                cv2.putText(frame, "No hand detected", (16, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (80, 80, 255), 2, cv2.LINE_AA)

            cv2.imshow("Hand Mouse", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):   # ESC or Q
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

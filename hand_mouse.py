import cv2
import mediapipe as mp
import pyautogui
import math

screen_w, screen_h = pyautogui.size()

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

clicked = False  

with HandLandmarker.create_from_options(options) as landmarker:

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = landmarker.detect_for_video(mp_image, int(cv2.getTickCount()))

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            #  Move mouse (index finger)
            x = hand[8].x
            y = hand[8].y

            mouse_x = int(x * screen_w)
            mouse_y = int(y * screen_h)

            pyautogui.moveTo(mouse_x, mouse_y)

            #  Thumb click
            thumb = hand[4]
            index = hand[8]

            distance = math.sqrt(
                (thumb.x - index.x) ** 2 +
                (thumb.y - index.y) ** 2
            )

            # click logic (debounce)
            if distance < 0.03 and not clicked:
                pyautogui.click()
                clicked = True

            elif distance >= 0.03:
                clicked = False

        cv2.imshow("AI Hand Mouse (Tasks)", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

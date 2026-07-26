import math
import time

import cv2
import mediapipe as mp
from pycaw.pycaw import AudioUtilities


cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

previous_time = time.time()
frame_count = 0
fps = 0

device = AudioUtilities.GetSpeakers()
volume = device.EndpointVolume

with mp_hands.Hands() as hands:
    while True:
        ret, frame = cam.read()

        if not ret:
            print("COULD NOT CAPTURE")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            h, w, c = frame.shape

            for hand_landmark in result.multi_hand_landmarks:
                landmarks = {}

                for landmark_id, landmark in enumerate(hand_landmark.landmark):
                    center_x = int(landmark.x * w)
                    center_y = int(landmark.y * h)
                    landmarks[landmark_id] = (center_x, center_y)

                if 4 in landmarks and 8 in landmarks:
                    x1, y1 = landmarks[4]
                    x2, y2 = landmarks[8]

                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    cv2.circle(frame, (x1, y1), 15, (0, 255, 255), -1)
                    cv2.circle(frame, (x2, y2), 15, (0, 255, 255), -1)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.circle(frame, (cx, cy), 15, (0, 255, 255), -1)

                    length = math.hypot(x2 - x1, y2 - y1)

                    max_distance = 400
                    length = max(0, min(length, max_distance))

                    if length < 47:
                        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
                        length = 0

                    volume_percent = (length / max_distance) * 100

                    max_volume_level = 65
                    volume_threshold = int((volume_percent / 100) * max_volume_level)
                    volume_amount = volume_threshold - max_volume_level

                    volume.SetMasterVolumeLevel(volume_amount, None)

                    bar_y = int(-3 * volume_percent + 500)

                    cv2.rectangle(frame, (47, 197), (103, 502), (0, 0, 255), 3)
                    cv2.rectangle(frame, (50, bar_y), (100, 499), (0, 255, 0), -1)

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmark,
                    mp_hands.HAND_CONNECTIONS
                )

        frame_count += 1
        current_time = time.time()

        if current_time - previous_time >= 1:
            fps = frame_count
            frame_count = 0
            previous_time = current_time

        cv2.putText(
            frame,
            str(fps),
            (20, 50),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (255, 0, 0),
            3
        )

        cv2.imshow("webCam", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quiting...")
            break

cam.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

mp_draw = mp.solutions.drawing_utils

previous_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print("COULD NOT CAPTURE")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmark.landmark):
                h ,w ,c = frame.shape
                center_x , center_y = int(landmark.x * w) , int(landmark.y * h)
                print(id ,center_x,center_y)
                cv2.circle(frame,(center_x,center_y),10,(255,255,0),-1)

            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    frame_count += 1
    current_time = time.time()

    if current_time - previous_time >= 1:
        fps = frame_count
        frame_count = 0
        previous_time = current_time

    cv2.putText(frame, str(fps), (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('webCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quiting...")
        break

cam.release()
cv2.destroyAllWindows()
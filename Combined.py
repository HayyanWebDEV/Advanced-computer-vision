import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
mp_Draw1 = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)

mp_face_detection = mp.solutions.face_detection
mp_Draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(0.5)

previous_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print("COULD NOT CAPTURE")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    result = hands.process(rgb_frame)
    results1 = face_mesh.process(rgb_frame)

    if results.detections:
        for id,detection in enumerate(results.detections):
            score = round(detection.score[0],2)
            bboxC = detection.location_data.relative_bounding_box
            h,w,c = frame.shape
            bbox= int(bboxC.xmin * w),int(bboxC.ymin * h),int(bboxC.width * w),int(bboxC.height * h)
            cv2.rectangle(frame,bbox,(0 ,255, 0),3)
            cv2.putText(frame , f"Face: {int(score * 100)}%" , (bbox[0] , bbox[1] - 20) ,
                        cv2.FONT_HERSHEY_PLAIN , 3 , (255 , 0 , 0) , 3)

    if result.multi_hand_landmarks:
        for hand_landmark in result.multi_hand_landmarks:
            for id, landmark in enumerate(hand_landmark.landmark):
                h ,w ,c = frame.shape
                center_x , center_y = int(landmark.x * w) , int(landmark.y * h)
                print(id ,center_x,center_y)
                cv2.circle(frame,(center_x,center_y),10,(255,255,0),-1)

            mp_draw.draw_landmarks(frame, hand_landmark, mp_hands.HAND_CONNECTIONS)

    if results1.multi_face_landmarks:
        for face_landmark in results1.multi_face_landmarks:
            mp_Draw1.draw_landmarks(frame,face_landmark,mp_face_mesh.FACEMESH_TESSELATION)

    frame_count += 1
    current_time = time.time()

    if current_time - previous_time >= 1:
        fps = frame_count
        frame_count = 0
        previous_time = current_time

    cv2.putText(frame,f"Fps: {int(fps)}", (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('webCam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quiting...")
        break

cam.release()
cv2.destroyAllWindows()
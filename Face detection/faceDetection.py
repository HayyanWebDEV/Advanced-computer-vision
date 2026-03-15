import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)

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

    if results.detections:
        for id,detection in enumerate(results.detections):
            score = round(detection.score[0],2)
            bboxC = detection.location_data.relative_bounding_box
            h,w,c = frame.shape
            bbox = int(bboxC.xmin * w),int(bboxC.ymin * h),int(bboxC.width * w),int(bboxC.height * h)
            cv2.rectangle(frame,bbox,(0 ,255, 0),3)
            cv2.putText(frame , f"Face: {int(score * 100)}%" , (bbox[0] , bbox[1] - 20) ,
                        cv2.FONT_HERSHEY_PLAIN , 3 , (255 , 0 , 0) , 3)

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
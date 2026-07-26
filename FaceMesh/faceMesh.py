import cv2
import mediapipe as mp
import time

cam = cv2.VideoCapture(0)

mp_face_mesh = mp.solutions.face_mesh
mp_Draw = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2)

previous_time = time.time()
frame_count = 0
fps = 0

while True:
    ret, frame = cam.read()

    if not ret:
        print("COULD NOT CAPTURE")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmark in results.multi_face_landmarks:
            mp_Draw.draw_landmarks(frame,face_landmark,mp_face_mesh.FACEMESH_TESSELATION)

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
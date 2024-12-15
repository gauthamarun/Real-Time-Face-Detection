import threading
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_img = cv2.imread("reference.jpeg")
lock = threading.Lock()  # To avoid race conditions

def checkface(frame):
    global face_match
    try:
        result = DeepFace.verify(frame, reference_img.copy())
        with lock:
            face_match = result['verified']
    except Exception as e:
        print(f"Error in DeepFace verification: {e}")
        with lock:
            face_match = False

while True:
    ret, frame = cap.read()
    if ret:
        if counter % 30 == 0:  # Process every 30 frames
            threading.Thread(target=checkface, args=(frame.copy(),)).start()
        counter += 1

        with lock:
            if face_match:
                cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

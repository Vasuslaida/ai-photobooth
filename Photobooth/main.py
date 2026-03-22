import cv2
from filters import apply_filter
from face_detection import detect_faces_and_smiles

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

filter_mode = "normal"
count = 0

import time
last_capture_time = 0

while True:
    ret, frame = cap.read()
    original_frame = frame.copy()

    if not ret:
        break

    key = cv2.waitKey(1)

    if key == ord('n'):
        filter_mode = "normal"
    elif key == ord('g'):
        filter_mode = "gray"
    elif key == ord('b'):
        filter_mode = "blur"
    elif key == ord('r'):
        filter_mode = "red"
    elif key == ord('c'):
        filter_mode = "cartoon"
    elif key == ord('q'):
        break

    frame = apply_filter(frame, filter_mode)
    faces, smile_detected = detect_faces_and_smiles(original_frame)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    current_time = time.time()

    if smile_detected and current_time - last_capture_time > 3:
        filename = f"smile_capture_{count}.jpg"
        cv2.imwrite(filename, original_frame)
        print("Smile detected! Photo captured.")
        count += 1
        last_capture_time = current_time
    cv2.putText(frame, f"Faces: {len(faces)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

    cv2.imshow("Photobooth", frame)

    if key == ord('s'):
        filename = f"photo_{count}.jpg"
        cv2.imwrite(filename, frame)
        count += 1

cap.release()
cv2.destroyAllWindows()
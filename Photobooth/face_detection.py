import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
    
smile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_smile.xml"
)

def detect_faces_and_smiles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    smiles_detected = False

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]

        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.8,
            minNeighbors=20
        )

        if len(smiles) > 0:
            smiles_detected = True

    return faces, smiles_detected
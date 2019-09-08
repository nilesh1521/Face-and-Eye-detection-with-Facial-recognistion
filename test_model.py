import cv2
import numpy as np
import pickle


img = cv2.imread("faces.jpeg",1)
Path = "haarcascade_frontalface_default.xml"
path = "haarcascade_eye.xml"

faceCascade = cv2.CascadeClassifier(Path)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name":1}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
eye_cascade = cv2.CascadeClassifier(path)

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30))
    eyes = eye_cascade.detectMultiScale(gray,scaleFactor=1.05,minNeighbors=5,minSize=(40, 40))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = ( 255,0,0)
            stroke = 2
            cv2.putText(frame , name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

    for (a, b, c, d) in eyes:
        cv2.rectangle(frame, (a, b), (a+c, b+d), (0, 0, 255), 2)


    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
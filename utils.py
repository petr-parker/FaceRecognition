import cv2
from deepface import DeepFace
import dlib
import numpy as np
import os

def match(points):
    for filename in os.listdir('database'):
        loaded_array = np.load('database/' + filename, allow_pickle=True)
        distance = sum([p.x**2 + p.y**2 for p in (loaded_array - np.array(points))])
        print(distance)
        if distance < 15:
            return filename.split('.')[0]
    return "Unknown"

def add_person(images, label):
    face_detector = dlib.get_frontal_face_detector() # распознает лица
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # model_path = "shape_predictor_68_face_landmarks.dat"
    # align = openface.AlignDlib(model_path) # приводит модель к нужному виду

    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rectangles = face_detector(gray, 1)

        assert len(rectangles) == 1
        rectangle = rectangles[0]

        x, y, w, h = rectangle.left(), rectangle.top(), rectangle.right() - rectangle.left(), rectangle.bottom() - rectangle.top()

        shape = shape_predictor(gray, rectangle)
        points_array = np.array(shape.parts())
        np.save(f"database/{label}.npy", points_array)

        font = cv2.FONT_HERSHEY_SIMPLEX
        color, thick = (255, 0, 0), 5

        cv2.rectangle(image, (x, y), (x + w, y + h), color, thick)
        cv2.putText(image, label, (x, y + h + 40), font, 1, color, 2)
        cv2.imshow("Display window", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def detect_persons(video):
    cap = cv2.VideoCapture(video)
    color, thick = (255, 0, 0), 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    face_detector = dlib.get_frontal_face_detector() # распознает лица
    shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangles = face_detector(gray, 1)


        for rectangle in rectangles:
            x, y, w, h = rectangle.left(), rectangle.top(), rectangle.right() - rectangle.left(), rectangle.bottom() - rectangle.top()
            shape = shape_predictor(gray, rectangle)
            points_array = shape.parts()

            label = match(points_array)

            font = cv2.FONT_HERSHEY_SIMPLEX
            color, thick = (255, 0, 0), 5

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thick)
            cv2.putText(frame, label, (x, y + h + 40), font, 1, color, 2)

        cv2.imshow('Video', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

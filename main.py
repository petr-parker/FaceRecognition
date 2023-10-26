import cv2
import utils
import numpy as np
from FaceDetector import *
from FaceRecognition import *
from VecComparator import *

def show_image(img, faces, labels):
    color, thick = (0, 255, 0), 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face, label in zip(faces, labels):
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        cv2.putText(img, label, (x, y + h + 40), font, 1, color, 2)
    cv2.imshow("Display window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_frame(img, faces, labels):
    color, thick = (255, 0, 0), 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face, label in zip(faces, labels):
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        cv2.putText(img, label, (x, y + h + 40), font, 1, color, 2)
    cv2.imshow("Frame", img)

def add_person(img: np.array, detector: FaceDetector, recognizer: FaceRecognition, id):
    faces = detector.detect(img)
    features = recognizer.extract(img, faces)
    show_image(img, faces, [id])
    assert len(features) == 1
    np.save(f"database/{id}.npy", features[0])


def read_capture(cap, detector, recognizer, comporator):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect(frame)
        features = recognizer.extract(frame, faces)
        ids = comporator.compare(features)
        show_frame(frame, faces, ids)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    detector_config = {}
    detector = DlibDetector(detector_config)

    recognizer_config = {'shape_predictor' : 'shape_predictor_68_face_landmarks.dat'}
    recognizer = DlibRecognition(recognizer_config)

    comporator_config = {'database' : 'database'}
    comparator = EuclidianComparator(comporator_config)

    # image = cv2.imread('media/my_face.jpg')
    # add_person(image, detector, recognizer, 'Petr')

    cap = cv2.VideoCapture('media/my_face.mp4')
    # cap = cv2.VideoCapture(0)
    read_capture(cap, detector, recognizer, comparator)


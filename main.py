import cv2
import utils
import numpy as np
from FaceDetector import *
from FaceRecognition import *
from VecComparator import *


def add_person(img: np.array, detector: FaceDetector, recognizer: FaceRecognition, id):
    faces = detector.detect(img)
    features = recognizer.extract(img, faces)
    assert len(features) == 1
    np.save(f"database/{id}.npy", features[0])





if __name__ == '__main__':
    detector_config = {}
    detector = DlibDetector(detector_config)

    recognizer_config = {'shape_predictor' : 'shape_predictor_68_face_landmarks.dat'}
    recognizer = DlibRecognition(recognizer_config)

    image = cv2.imread('media/my_face.jpg')
    add_person(image, detector, recognizer, 'Petr')





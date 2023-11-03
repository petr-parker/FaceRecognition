import cv2
import numpy as np
from FaceDetector import *
from FaceRecognizer import *
from VecComparator import *

def show_frame(img, faces, labels):
    color, thick = (255, 0, 0), 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face, label in zip(faces, labels):
        x, y, w, h = face
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        cv2.putText(img, label, (x, y + h + 40), font, 1, color, 2)
    cv2.imshow("Frame", img)

def add_person(img: np.array, detector: FaceDetector, recognizer: FaceRecognizer, id):
    faces = detector.detect(img)
    features = recognizer.extract(img, faces)
    assert len(features) == 1
    show_frame(img, faces, [id])
    np.save(f"database/{id}.npy", features[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image(detector, recognizer, comporator, image):
    faces = detector.detect(image)
    features = recognizer.extract(image, faces)
    ids, confs = comporator.compare(features)
    labels = [f"{id}: {int(conf * 100)}%" for id, conf in zip(ids, confs)]
    show_frame(image, faces, labels)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_capture(detector, recognizer, comporator, cap=None):
    if cap is None:
        cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect(frame)
        features = recognizer.extract(frame, faces)
        ids, confs = comporator.compare(features)
        labels = [f"{id}: {int(conf * 100)}%" for (id, conf) in zip(ids, confs)]
        show_frame(frame, faces, labels)
        if cv2.waitKey(1) == ord('q'):
            break
        if cv2.waitKey(1) == ord('s'):
            id = 'Kate'
            np.save(f"database/{id}.npy", features[0])
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detector_config = {}
    detector = DlibDetector(detector_config)

    # recognizer_config = {'model' : 'models/shape_predictor_68_face_landmarks.dat' }
    # recognizer = DlibRecognizer(recognizer_config)
    recognizer_config = {'model' : 'models/face_recognition_sface_2021dec.onnx' }
    recognizer = OpenCVdnnRecognizer(recognizer_config)
    
    comporator_config = {'database' : 'database', 'threshold' : 1.128}
    comparator = EuclidianComparator(comporator_config)

    # image = cv2.imread('media/Shaldon.jpg')
    # add_person(image, detector, recognizer, 'Shaldon')

    # image = cv2.imread('media/my_face.jpg')
    # process_image(detector, recognizer, comparator, image)

    cap = cv2.VideoCapture('media/my_face.mp4')
    process_capture(detector, recognizer, comparator)


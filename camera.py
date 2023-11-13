import json
import sys
import cv2
import numpy as np
import datetime
import psycopg2
import FaceDetector
import FaceRecognizer
import VecComparator
from base64 import b16encode as enc16

def initialize_detector(config):
    detector_config = {
        'model' : 'models/face_detection_yunet_2023mar.onnx',
        'inputsize' : (320, 320),
        'confthreshol' : 0.9,
        'nmsthreshold' : 0.3,
        'topk' : 1000,
        'backendid' : cv2.dnn.DNN_BACKEND_OPENCV,
        'targetid' : cv2.dnn.DNN_TARGET_CPU,
    }
    return FaceDetector.YunetDetector(detector_config)

def initialize_recognizer(config):
    nets_decode = {
        'sface' : 'models/face_recognition_sface_2021dec.onnx',
    }
    recognizer_config = { 'model' : nets_decode[config['net_name']] }
    return FaceRecognizer.OpenCVdnnRecognizer(recognizer_config)

def initialize_comparator(config):
    comporator_config = {'threshold' : 9}
    return VecComparator.EuclidianComparator(comporator_config)

def read_data():
    conn = psycopg2.connect(dbname='facerecognition', user='postgres', password='1237', host='localhost')
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM employees')
        tuples = cursor.fetchall()
    conn.close() # закрываем соединение
    data = []
    for tup in tuples:
        row = {
            'id_employee' : tup[0],
            'firstname' : tup[1],
            'lastname' : tup[3],
            'sface' : tup[9]['data'],
        }
        data.append(row)

    return data

def show_frame(img, faces, labels):
    color_success, color_failure, thick = (0, 255, 0), (255, 0, 0), 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    for face, label in zip(faces, labels):
        x, y, w, h = [int(el) for el in face[:4]]
        # for i in range(5):
        #     x_circle, y_circle = int(face[4 + i*2]), int(face[5 + i*2])
        #     cv2.circle(img, (x_circle, y_circle), 3, color, -1)
        color = color_failure if label == '' else color_success
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thick)
        cv2.putText(img, label, (x, y + h + 40), font, 1, color, 2)
    cv2.imshow("Recognize", img)

def video(data, detector, recognizer, comporator):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        faces = detector.detect(frame)
        features = recognizer.extract(frame, faces)
        ids, labels, confs = comporator.compare(features, data)
        show_frame(frame, faces, labels)

        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    config = {'net_name' : 'sface'}

    detector = initialize_detector(config)
    recognizer = initialize_recognizer(config)
    comparator = initialize_comparator(config)

    data = read_data()
    print(len(data))

    video(data, detector, recognizer, comparator)




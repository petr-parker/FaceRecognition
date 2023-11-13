import json
import sys
import cv2
import numpy as np
import datetime
import psycopg2
import FaceDetector
import FaceRecognizer
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

def add_person(config, features):
    try:
        with open(config['photo_path'], 'rb') as f:
            photo_binary = enc16(f.read())
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        features = [list(feature) for feature in features]
        
        values = "\', \'".join(['\'' + config['firstname'], config['middlename'], config['lastname'] + '\', ']) + '\'\\x' + str(photo_binary)[2:] + ', ' + \
            "\', \'".join(['\'' + config['birth_date'], config['job_name'], config['phone_number'], current_time, '{ "data" : ' + str(list(features)) + ' }\''])

        conn = psycopg2.connect(dbname='facerecognition', user='postgres', password='1237', host='localhost')
        cur = conn.cursor()
        cur.execute('INSERT INTO employees(firstname, middlename, lastname, photo_bin, birth_date, job_name, phone_number, creation_time, sface) VALUES (' + values + ');')
        cur.close()
        conn.commit()
        conn.close() # закрываем соединение
        print('Успешная запись')
        return True
    except:
        print('Запись не удалась')
        return False

if __name__ == '__main__':
    config_path = sys.argv[1]

    with open(config_path, "r") as jsonfile:
        config = json.load(jsonfile)

    detector = initialize_detector(config)
    recognizer = initialize_recognizer(config)

    img = cv2.imread(config['photo_path'])
    faces = detector.detect(img)
    assert len(faces) == 1
    features = recognizer.extract(img, faces)

    add_person(config, features[0])




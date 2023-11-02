import numpy as np
import dlib
import cv2
# from tensorflow.python.platform import gfile
# import tensorflow.compat.v1 as tf



class FaceRecognizer:
    def __init__(self, config: dict) -> bool:
        pass
    def extract(self, img: np.array, faces: list) -> list:
        pass

class DlibRecognizer(FaceRecognizer):
    def __init__(self, config: dict) -> bool:
        self.shape_predictor = dlib.shape_predictor(config['model'])
    def extract(self, img: np.array, faces: list) -> list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []
        for face in faces:
            rectangle = dlib.rectangle(left=face[0], top=face[1], right=face[0]+face[2], bottom=face[1]+face[3])
            shape = self.shape_predictor(gray, rectangle)
            points_list = shape.parts()
            x_min = min([p.x for p in points_list])
            x_max = max([p.x for p in points_list])
            y_min = min([p.y for p in points_list])
            y_max = max([p.y for p in points_list])
            x_middle = (x_min + x_max) / 2
            y_middle = (y_min + y_max) / 2
            width = x_max - x_min
            height = y_max - y_min
            features_array = np.array([[(p.x - x_middle) / width, (p.y - y_middle) / height] for p in points_list])
            features.append(features_array)
        return features
    
class OpenCVdnnRecognizer(FaceRecognizer):
    def __init__(self, config: dict) -> bool:
        self.net = cv2.FaceRecognizerSF.create(config['model'], "")
    
    def extract(self, img: np.array, faces: list) -> list:

        self.blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []
        for face in faces:
            face1_align = self.recognizer.alignCrop(gray, faces1[1][0])
            face1_align = self.net.alignCrop(img1, faces1[1][0])
            features_array = self.net.feature(face1_align)
            features.append(features_array)
        return features

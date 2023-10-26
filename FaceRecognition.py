import numpy as np
import dlib
import cv2

class FaceRecognition:
    def __init__(self, config: dict) -> bool:
        pass
    def extract(self, img: np.array, faces: list) -> list:
        pass

class DlibRecognition(FaceRecognition):
    def __init__(self, config: dict) -> bool:
        self.shape_predictor = dlib.shape_predictor(config['shape_predictor'])
    def extract(self, img: np.array, faces: list) -> list:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []
        for face in faces:
            rectangle = dlib.rectangle(left=face[0], top=face[1], right=face[0]+face[2], bottom=face[1]+face[3])
            shape = self.shape_predictor(gray, rectangle)
            points_list = shape.parts()
            features_array = np.array([[p.x, p.y] for p in points_list])
            features.append(features_array)
        return features

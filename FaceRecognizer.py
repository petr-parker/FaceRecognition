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
            features_array = np.array([[p.x - face[0], p.y - face[1]] for p in points_list])
            features.append(features_array)
        return features
    
# class OpenCVdnnRecognizer(FaceRecognizer):
#     def __init__(self, config: dict) -> bool:
#         self.net = cv2.dnn.readNetFromCaffe(config['prototxt'], config['model'])
    
#     def load_model():
#         v = ftk.Verification()
#         v.load_model("./Models/FaceDetection/")
#         v.initial_input_output_tensors()
#         return v
    
#     def extract(self, img: np.array, faces: list) -> list:

#         self.blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#         features = []
#         for face in faces:
#             features_array = FaceDetection.v.img_to_encoding(cv2.resize(face, (160, 160)), FaceDetection.image_size)
#             features.append(features_array)
#         return features

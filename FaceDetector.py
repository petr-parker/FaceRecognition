import numpy as np
import dlib
import cv2

class FaceDetector:
    def __init__(self, config: dict) -> bool:
        pass
    def detect(self, img: np.array) -> list:
        pass

class DlibDetector(FaceDetector):
    def __init__(self, config: dict) -> bool:
        self.face_detector = dlib.get_frontal_face_detector()

    def detect(self, img: np.array) -> list:
        '''
        :param img: image
        '''
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rectangles = self.face_detector(gray, 1)
        faces = []
        for rectangle in rectangles:
            x, y, w, h = rectangle.left(), rectangle.top(), rectangle.right() - rectangle.left(), rectangle.bottom() - rectangle.top()
            faces.append((x, y, w, h))
        return faces


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

class YunetDetector(FaceDetector):
    def __init__(self, config: dict) -> bool:
        self._modelPath = config['model']
        self._inputSize = config['inputsize']
        self._confThreshold = config['confthreshol']
        self._nmsThreshold = config['nmsthreshold']
        self._topK = config['topk']
        self._backendId = config['backendid']
        self._targetId = config['targetid']

        self._model = cv2.FaceDetectorYN.create(
            model=self._modelPath,
            config="",
            input_size=self._inputSize,
            score_threshold=self._confThreshold,
            nms_threshold=self._nmsThreshold,
            top_k=self._topK,
            backend_id=self._backendId,
            target_id=self._targetId)

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def detect(self, image):
        h, w, _ = image.shape
        self.setInputSize([w, h])
        faces = self._model.detect(image)
        if faces[1] is None:
            return []
        else:
            return faces[1]

import os
import numpy as np

class VecComparotor:
    def __init__(self, config: dict) -> bool:
        self.threshold = config['threshold']
    def dist(self, vec1: np.array, vec2: np.array) -> float:
        raise NotImplementedError
    def compare(self, features, data) -> list:
        '''
        Возвращает списки id, confidence (-1, если не найден)
        '''
        ids, labels, confidences = [], [], []
        for feature in features:
            chosen_id = "-1"
            chosen_label = ""
            confidence = 0 
            for row in data:
                other = np.array(row['sface'])
                distance = self.dist(other, features)
                if distance < self.threshold:
                    chosen_id = row['id_employee']
                    chosen_label = row['firstname'] + row['lastname']
                    confidence = (self.threshold - distance) / self.threshold
            ids.append(chosen_id)
            labels.append(chosen_label)
            confidences.append(confidence)
        return ids, labels, confidences

class EuclidianComparator(VecComparotor):
    def __init__(self, config: dict) -> bool:
        super().__init__(config)
    def dist(self, vec1: np.array, vec2: np.array) -> float:
        return np.linalg.norm(vec1 - vec2)
            
class ConeComparator(VecComparotor):
    def __init__(self, config: dict) -> bool:
        super().__init__(config)
    def compare(self, vectors: list) -> list:
        '''
        Возвращает списки id, confidence (-1, если не найден)
        '''
        ids = []
        for vector in vectors:
            chosen_id = "-1"
            for id in os.listdir(self.database):
                loaded_array = np.load(self.database + '/' + id, allow_pickle=True)
                distance = self.dist(loaded_array, vector)
                np.linalg.norm(loaded_array - vector)
                if distance > self.threshold:
                    chosen_id = id.split('.')[0]
            ids.append(chosen_id)
        return ids
    def dist(self, vec1: np.array, vec2: np.array) -> float:
        dot = np.sum(vec1 * vec2)
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        return dot / (vec1_norm * vec2_norm)





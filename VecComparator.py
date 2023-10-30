import os
import numpy as np

class VecComparotor:
    def __init__(self, config: dict) -> bool:
        self.database = config['database']
        self.threshold = config['threshold']
    def dist(self, vec1: np.array, vec2: np.array) -> float:
        pass
    def compare(self, vectors: list) -> list:
        '''
        Возвращает список id (-1, если не найден)
        '''
        ids = []
        for vector in vectors:
            chosen_id = "-1"
            for id in os.listdir(self.database):
                loaded_array = np.load(self.database + '/' + id, allow_pickle=True)
                distance = self.dist(loaded_array, vector)
                np.linalg.norm(loaded_array - vector)
                print(distance)
                if distance < self.threshold:
                    chosen_id = id
            ids.append(chosen_id)
        return ids

class EuclidianComparator(VecComparotor):
    def __init__(self, config: dict) -> bool:
        super().__init__(config)
    def dist(self, vec1: np.array, vec2: np.array) -> float:
        return np.linalg.norm(vec1 - vec2)
            
class ConeComparator(VecComparotor):
    def __init__(self, config: dict) -> bool:
        super().__init__(config)
    def dist(self, vec1: np.array, vec2: np.array) -> float:
        dot = np.sum(vec1 * vec2)
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        return (2 - (1 + dot / (vec1_norm * vec2_norm)))





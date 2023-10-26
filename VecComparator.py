import os
import numpy as np

class VecComparotor:
    def __init__(self, config: dict) -> bool:
        pass
    def compare(self, vectors: list) -> list:
        pass

class EuclidianComparator(VecComparotor):
    def __init__(self, config: dict) -> bool:
        self.database = config['database']
    def compare(self, vectors: list) -> list:
        '''
        Возвращает список id (-1, если не найден)
        '''
        ids = []
        for vector in vectors:
            chosen_id = "-1"
            for id in os.listdir(self.database):
                loaded_array = np.load(self.database + '/' + id, allow_pickle=True)
                distance = np.linalg.norm(loaded_array - vector)
                print(distance)
                if distance < 15:
                    chosen_id = id
            ids.append(chosen_id)
        return ids
            






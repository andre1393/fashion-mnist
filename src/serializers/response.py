import numpy as np

from src.settings import CLASS_NAMES


class ResponseSerializer:
    def __init__(self, names=False):
        self.names = names

    def serialize(self, predictions):
        idx = [np.argmax(i) for i in predictions]
        if self.names:
            return [CLASS_NAMES[i] for i in idx]
        else:
            return idx

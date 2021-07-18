import numpy as np
from fastapi.responses import JSONResponse

from src.settings import CLASS_NAMES


class ResponseSerializer:
    def __init__(self, names=False):
        self.names = names

    def serialize(self, predictions):
        idx = [np.argmax(i) for i in predictions]
        data = [CLASS_NAMES[i] for i in idx] if self.names else idx
        return JSONResponse({"predictions": data})

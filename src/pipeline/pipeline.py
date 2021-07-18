from abc import ABC

from sklearn.pipeline import Pipeline

from src.pipeline.scale_image_values import ScaleImageValues
from src.pipeline.reshape_data import ReshapeData
from src.pipeline.predictor import Predictor
from src.settings import MODEL_PATH


class PredictPipeline(ABC):
    @staticmethod
    def get_pipeline():
        return Pipeline([
            ('reshape_data', ReshapeData()),
            ('scale_image_values', ScaleImageValues()),
            ('predictor', Predictor(MODEL_PATH))
        ])

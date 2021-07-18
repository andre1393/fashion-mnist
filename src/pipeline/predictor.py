import tensorflow as tf
from sklearn.base import BaseEstimator, ClassifierMixin

from src.utils import Singleton


class Predictor(BaseEstimator, ClassifierMixin, metaclass=Singleton):

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = tf.keras.models.load_model(self.model_path)

    def fit(self, x, y):
        return self

    def predict(self, data):
        return data, self.model.predict(data)

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator


class ReshapeData(BaseEstimator, TransformerMixin):
    def fit(self, y=None):
        return self

    @staticmethod
    def transform(data):
        array_data = np.array(data)
        return array_data.reshape(*array_data.shape, 1)

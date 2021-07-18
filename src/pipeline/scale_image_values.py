from sklearn.base import TransformerMixin, BaseEstimator


class ScaleImageValues(BaseEstimator, TransformerMixin):
    def fit(self, y=None):
        return self

    @staticmethod
    def transform(data):
        return data/255

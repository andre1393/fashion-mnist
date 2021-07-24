import numpy as np

from src.pipeline.predictor import Predictor


def test_pipeline():
    x = np.load('tests/data/x_processed.npy')
    y = np.load('tests/data/predictions.npy')

    predictor = Predictor('models/tf_cnn')
    _, predictions = predictor.predict(x.reshape(*x.shape, 1))
    idx = [np.argmax(x) for x in predictions]

    assert (idx == y).all()

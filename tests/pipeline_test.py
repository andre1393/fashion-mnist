import numpy as np

from src.pipeline.pipeline import PredictPipeline


def test_pipeline():
    x = np.load('tests/data/x_raw.npy')
    y = np.load('tests/data/predictions.npy')

    pipeline = PredictPipeline().get_pipeline()
    _, predictions = pipeline.predict(x)
    idx = [np.argmax(x) for x in predictions]

    assert (idx == y).all()

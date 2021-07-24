import os
import argparse
import logging
import numpy as np

import mlflow
from tensorflow.keras import models, layers

from utils import load_npy_files, save_tf_model, save_npy_files

logger = logging.getLogger(__name__)


def build_cnn():
    """
    Modelling - Model on CNN

    :return: CNN model architecture
    """
    # create a sequential model i.e. empty neural network which has no layers in it.
    model = models.Sequential()

    # ==================== Feature Detection / extraction Block ====================#

    # Add first convolutional block - To deal with images we use Conv2D and for colour images and shape use Conv3D
    # in the first block we need to mention input_shape
    model.add(layers.Conv2D(6, (3, 3), input_shape=(28, 28, 1), activation='relu'))
    # Add the max pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Add Second convolutional block
    model.add(layers.Conv2D(10, (3, 3), activation='relu'))
    # Add the max pooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # ==================== Transition Block (from feature detection to classification) ====================#

    # Add Flatten layer. Flatten simply converts metrics to array
    model.add(layers.Flatten(input_shape=(28, 28)))  # this will flatten the image and after this Classification happens

    # ==================== Classification Block ====================#

    # Classification segment - fully connected network
    # The Dense layer does classification and is deep neural network. Dense layer always accept the array.
    model.add(layers.Dense(128, activation='relu'))  # as C5 layer in above image.
    # this 120 is hyper parameter which is number of neuron

    # Add the output layer
    # as Output layer in above image. The output layer normally have softmax activation
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    return model


def reshape_features(x_train, x_test):
    """
    reshape features

    :param x_train: x_train dataset
    :param x_test: x_test dataset
    :return: x_train and x_test reshaped
    """
    return x_train.reshape(60000, 28, 28, 1), x_test.reshape(10000, 28, 28, 1)


def train_model(model, x_train, y_train, x_test, y_test):
    """
    Train the model

    :return: trained model
    """
    model.fit(x_train, y_train, epochs=20, batch_size=1000, verbose=True, validation_data=(x_test, y_test))
    return model


def main(input_path, model_path, model_name, test_size):
    x_train, y_train, x_test, y_test = load_npy_files(input_path, ['x_train', 'y_train', 'x_test', 'y_test'])
    x_train_reshaped, x_test_reshaped = reshape_features(x_train, x_test)

    mlflow.tensorflow.autolog()
    model = train_model(build_cnn(), x_train_reshaped, y_train, x_test_reshaped, y_test)
    save_npy_files('tests/data/', [[np.argmax(x) for x in model.predict(x_test_reshaped)][:test_size]], ['predictions'])
    saved_model_name = os.path.join(model_path, model_name)
    save_tf_model(model, saved_model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='path where datasets are loaded from', default='data/processed')
    parser.add_argument('--model-path', help='path where trained model is saved', default='models')
    parser.add_argument('--model-name', help='model name', default='tf_cnn')
    parser.add_argument('--test-size', help='test dataset size for unit test', default=10)

    args = parser.parse_args()
    main(**vars(args))

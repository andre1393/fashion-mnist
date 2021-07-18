"""
Utils functions
"""
import os
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


def load_npy_files(input_path, files_prefixes):
    """
    Load `files_prefixes` numpy files from `input_path` and yield them.

    :param input_path: path where datasets are loaded from
    :param files_prefixes: list of numpy file names to load (without extension)
    :return: numpy objects loaded
    """
    for item in files_prefixes:
        with open(os.path.join(input_path, f'{item}.npy'), 'rb') as file:
            logger.info(f'loading {item}.npy file')
            yield np.load(file)


def save_npy_files(output_path, np_objs, filenames):
    """
    Save `np_objs` numpy objects into `output_path` with `filenames`.

    :param output_path: path where numpy objects are saved
    :param np_objs: list of numpy objects
    :param filenames: list of file names
    :return:
    """
    for item in zip(filenames, np_objs):
        with open(os.path.join(output_path, f'{item[0]}.npy'), 'wb') as file:
            logger.info(f'saving {item[0]} into {file.name}')
            np.save(file, item[1])


def save_tf_model(model, output_path):
    """
    Saves a tensorflow model into `output_path`
    :param model: tensorflow model
    :param output_path: path where the model is saved
    :return: None
    """
    model.save(output_path)


def load_tf_model(model_path):
    """
    Loads a tensorflow model from `model_path`
    :param model_path: path where the tensorflow model is loaded from
    :return: tensorflow model
    """
    return tf.keras.models.load_model(model_path)

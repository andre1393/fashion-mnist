"""
Load fashion mnist dataset from Zalando using keras lib and save into .npy files
"""
import argparse
import logging

import tensorflow as tf

from utils import save_npy_files

logger = logging.getLogger(__name__)


def main(output_path):
    """
    Saves fashion_mnist datasets into `data_path`

    :param output_path: path where datasets are saved
    :return: None
    """

    logger.info('loading datasets...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    save_npy_files(output_path, [x_train, y_train, x_test, y_test], ['x_train', 'y_train', 'x_test', 'y_test'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-path', help='path where raw datasets are saved.', default='data/raw')
    args = parser.parse_args()
    main(args.output_path)

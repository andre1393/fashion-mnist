"""
Load fashion mnist dataset from Zalando using keras lib and save into .npy files
"""
import os
import argparse
import logging

from tensorflow.keras.utils import to_categorical

from utils import load_npy_files, save_npy_files

logging.basicConfig(level=os.getenv('LOGLEVEL', 'INFO'))
logger = logging.getLogger(__name__)


def scale_image_values(x_train, x_test):
    """
    Pixel value of the image falls between 0 to 255.

    :return: x_train and y_train rescaled
    """

    x_train = x_train/255  # So, we are scale the value between 0 to 1 before by dividing each value by 255
    logger.info(f'x_train shape: {x_train.shape}')

    x_test = x_test/255  # So, we are scale the value between 0 to 1 before by dividing each value by 255
    logger.info(f'x_test shape: {x_test.shape}')

    return x_train, x_test


def one_hot_encoding(y_train, y_test):
    """ One hot encoding of the labels.

    :param y_train: y_train numpy object
    :param y_test: t_test numpy object
    :return: y_train and y_test with ohe
    """

    # Before one hot encoding
    logger.info(f'y_train Shape before ohe: {y_train.shape}')
    logger.info(f'y_test Shape before ohe: {y_test.shape}')

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # After one hot encoding
    logger.info(f'y_train Shape after ohe: {y_train.shape}')
    logger.info(f'y_test Shape after ohe: {y_test.shape}')

    return y_train, y_test


def main(input_path, output_path, test_size):
    """
    Saves fashion_mnist datasets into `data_path`

    :param input_path: path where data are read from.
    :param output_path: path where raw datasets are saved.
    :param test_size: test dataset size for unit test
    :return: None
    """

    logger.info(f'loading raw datasets from {input_path}...')
    x_train, y_train, x_test, y_test = load_npy_files(input_path, ['x_train', 'y_train', 'x_test', 'y_test'])
    # dataset for testing
    save_npy_files('tests/data', [x_test[:test_size]], ['x_raw'])

    x_train, x_test = scale_image_values(x_train, x_test)
    y_train, y_test = one_hot_encoding(y_train, y_test)

    logger.info(f'saving processed datasets into {output_path}')
    save_npy_files(output_path, [x_train, y_train, x_test, y_test], ['x_train', 'y_train', 'x_test', 'y_test'])
    save_npy_files('tests/data', [x_test[:test_size]], ['x_processed'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', help='path where data are read from.', default='data/raw')
    parser.add_argument('--output-path', help='path where raw datasets are saved.', default='data/processed')
    parser.add_argument('--test-size', help='test dataset size for unit test', default=10)
    args = parser.parse_args()
    main(args.input_path, args.output_path, args.test_size)

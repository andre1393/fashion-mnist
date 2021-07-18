import os

MODEL_PATH = 'models/tf_cnn'
CLASS_NAMES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
DB_SETTINGS = {
    'host': os.getenv('DB_HOST', 'mongo'),
    'port': os.getenv('DB_PORT', 27017),
    'username': os.getenv('DB_USERNAME'),
    'password': os.getenv('DB_PASSWORD')
}

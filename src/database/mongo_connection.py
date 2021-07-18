import logging
from pymongo import MongoClient

from src.utils import Singleton

logger = logging.getLogger(__name__)


class DBConnection(metaclass=Singleton):

    def __init__(self, db_settings):
        self.db_settings = db_settings
        self.client = MongoClient(
            host=db_settings['host'],
            port=int(db_settings['port']),
            username=db_settings['username'],
            password=db_settings['password'],
            authSource='admin'
        )

    def _get_database(self):
        return self.client['fashion-mnist']

    def save_request_info(self, request_id, raw, processed, predictions):
        db = self._get_database()
        logger.info(f'saving raw_data, processed_data and predictions for {len(raw)} predictions')
        db['predictions'].insert_many([
            {
                'request_id': request_id,
                'raw_data': raw[i],
                'processed_data': processed[i],
                'predictions': predictions[i]
            } for i in range(len(raw))]
        )




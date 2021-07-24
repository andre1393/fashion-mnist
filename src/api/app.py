import os
import logging
import uuid
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, Request

from src.pipeline.pipeline import PredictPipeline
from src.serializers.response import ResponseSerializer
from src.database.mongo_connection import DBConnection
from src.settings import DB_SETTINGS

logger = logging.getLogger(__name__)
app = FastAPI()


@app.get('/health', status_code=200)
def health_check():
    """application health check"""
    return JSONResponse({"status": "healthy"})


@app.post('/predict')
async def predict(request: Request):
    """
    Use input from request to transform data and make predictions, at the end parse prediction to class name
    :param request: API request object
    :return: predictions class names serialized as json
    """
    raw_data = (await request.json())['data']
    processed_data, predictions = PredictPipeline.get_pipeline().predict(raw_data)
    DBConnection(DB_SETTINGS).save_request_info(
        request.headers.get('request-id', str(uuid.uuid4())), raw_data, processed_data.tolist(), predictions.tolist()
    )
    return ResponseSerializer(names=True).serialize(predictions)


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host='127.0.0.1',
        port=os.environ.get('PORT', 8000),
        log_level=os.environ.get('LOGLEVEL', 'info').lower()
    )

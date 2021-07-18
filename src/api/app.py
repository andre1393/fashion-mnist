import os
import logging

from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, Request

from src.pipeline.pipeline import PredictPipeline
from src.serializers.response import ResponseSerializer

logger = logging.getLogger(__name__)
app = FastAPI()


@app.get('/health', status_code=200)
def health_check():
    return JSONResponse({"status": "healthy"})


@app.post('/predict')
async def predict(request: Request):
    body = await request.json()
    predicted = PredictPipeline.get_pipeline().predict(body['data'])
    predicted_processed = ResponseSerializer(names=True).serialize(predicted)
    return JSONResponse({"predictions": predicted_processed})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host='127.0.0.1',
        port=os.environ.get('PORT', 8000),
        log_level=os.environ.get('LOGLEVEL', 'info').lower()
    )

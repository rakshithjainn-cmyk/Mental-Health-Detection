from fastapi import FastAPI
from pydantic import BaseModel
from src.inference import predict

app = FastAPI(title='Mental Health Detection API')

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label_id: int
    label_name: str
    probs: list

@app.post('/predict', response_model=PredictionOut)
async def predict_endpoint(payload: TextIn):
    return predict(payload.text)

# Run with: uvicorn app.api:app --host 0.0.0.0 --port 8000

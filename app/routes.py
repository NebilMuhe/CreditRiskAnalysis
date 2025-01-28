from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils import load_model, preprocess_input, make_prediction
from app.config import Config

# Define a router for API endpoints
router = APIRouter()

# Load the trained model
model = load_model(model_path=Config.MODEL_PATH)

# Define input schema for API
class PredictionInput(BaseModel):
    transaction_id: str
    account_id: str
    amount: float
    currency_code: str
    product_category: str
    transaction_hour: int
    transaction_day: int
    transaction_month: int
    transaction_year: int
    recency: float
    frequency: float
    monetary: float

# Define output schema for API
class PredictionOutput(BaseModel):
    risk_label: str

# Endpoint for health check
@router.get("/health", status_code=200)
async def health_check():
    return {"status": "API is up and running"}

# Endpoint for predictions
@router.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput):
    try:
        # Preprocess the input
        features = preprocess_input(data)
        
        # Make prediction
        prediction = make_prediction(model, features)
        
        # Return prediction result
        return {"risk_label": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

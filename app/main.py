from fastapi import FastAPI
from app.routes import router

# Initialize FastAPI app
app = FastAPI(title="ML Model Serving API")

# Include API routes
app.include_router(router)
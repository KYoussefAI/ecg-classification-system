"""
ECG Cardiac Classification API
FastAPI backend for 12-lead ECG multi-label classification
"""

import logging
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from app.routes import auth, predict, history, stats
from app.services.model_service import ModelService
from app.database import init_db

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    await init_db()

    logger.info("Loading ECG model...")
    #ModelService.get_instance()

    logger.info("Model loaded. API ready.")
    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="CardioScan ECG Analysis API",
    description="AI-powered 12-lead ECG multi-label cardiac classification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(auth.router,    prefix="/api/auth",    tags=["Authentication"])
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(stats.router,   prefix="/api/stats",   tags=["Statistics"])


@app.get("/")
async def root():
    return {"message": "CardioScan API is running", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
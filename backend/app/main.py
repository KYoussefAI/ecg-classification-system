"""
ECG Cardiac Classification API
FastAPI backend for 12-lead ECG multi-label classification
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

from app.routes import auth, predict, history, stats
from app.services.model_service import ModelService
from app.database import init_db
from app.middleware import RequestSizeLimit

# Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    await init_db()

    logger.info("Loading ECG model...")
    ModelService.get_instance()

    logger.info("Model state: %s", ModelService.get_instance().state)
    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="CardioScan ECG Analysis API",
    description="AI-powered 12-lead ECG multi-label cardiac classification",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS — comma-separated origins in CORS_ORIGINS, or sensible dev defaults
_cors_raw = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
)
_allow_origins = [o.strip() for o in _cors_raw.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestSizeLimit)

# Routers
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(predict.router, prefix="/api/predict", tags=["Prediction"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(stats.router, prefix="/api/stats", tags=["Statistics"])


@app.get("/")
async def root():
    return {"message": "CardioScan API is running", "version": "1.0.0"}


@app.get("/health")
@app.get("/api/health")
async def health():
    from fastapi.responses import JSONResponse

    service = ModelService.get_instance()
    return JSONResponse(
        service.health(), status_code=200 if service.state == "ready" else 503
    )


@app.get("/api/model")
async def model_metadata():
    return ModelService.get_instance().public_metadata()

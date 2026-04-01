"""FastAPI application entry point for the Beer Tap Counter."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.database.connection import init_db
from backend.logging_config import setup_logging
from backend.routers.videos import router as videos_router
from backend.routers.counts import router as counts_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("Starting Beer Tap Counter API")
    init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(title="Beer Tap Counter API", lifespan=lifespan)


app.include_router(videos_router, prefix="/api/videos", tags=["videos"])
app.include_router(counts_router, prefix="/api/counts", tags=["counts"])


@app.get("/")
def root():
    return {"message": "Beer Tap Counter API", "status": "ok"}

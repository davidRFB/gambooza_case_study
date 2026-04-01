"""FastAPI application entry point for the Beer Tap Counter."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.database.connection import init_db
from backend.routers.videos import router as videos_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Beer Tap Counter API", lifespan=lifespan)


app.include_router(videos_router, prefix="/api/videos", tags=["videos"])


@app.get("/")
def root():
    return {"message": "Beer Tap Counter API", "status": "ok"}

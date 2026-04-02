"""SQLAlchemy ORM models for the Beer Tap Counter."""

from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)  # UUID name on disk
    original_name = Column(String, nullable=False)  # user's original filename
    upload_date = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # pending | processing | completed | error
    duration_sec = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)
    ml_approach = Column(String, nullable=True)
    processing_started_at = Column(DateTime, nullable=True)
    processing_finished_at = Column(DateTime, nullable=True)
    output_dir = Column(String, nullable=True)  # path to pipeline intermediate files
    restaurant_name = Column(String, nullable=True)  # e.g. "cerveceria_centro"
    camera_id = Column(String, nullable=True)  # e.g. "cam1"

    tap_events = relationship(
        "TapEvent",
        back_populates="video",
        cascade="all, delete-orphan",
    )


class TapEvent(Base):
    __tablename__ = "tap_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    tap = Column(String, nullable=False)  # "A" or "B"
    frame_start = Column(Integer, nullable=False)
    frame_end = Column(Integer, nullable=False)
    timestamp_start = Column(Float, nullable=False)  # seconds into video
    timestamp_end = Column(Float, nullable=False)
    confidence = Column(Float, nullable=True)
    count = Column(Integer, default=1)  # beers served in this event

    video = relationship("Video", back_populates="tap_events")

"""Count endpoints — query and summary."""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.database.models import TapEvent, Video
from backend.database.schemas import CountResult, CountSummary

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/", response_model=list[CountResult])
def get_counts(
    video_id: int | None = Query(None),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    tap: str | None = Query(None, pattern="^[AB]$"),
    db: Session = Depends(get_db),
):
    query = db.query(Video).filter(Video.status == "completed")
    if video_id:
        query = query.filter(Video.id == video_id)
    if date_from:
        query = query.filter(Video.upload_date >= date_from)
    if date_to:
        query = query.filter(Video.upload_date <= date_to)

    results = []
    for video in query.all():
        events_q = db.query(func.coalesce(func.sum(TapEvent.count), 0)).filter(
            TapEvent.video_id == video.id,
        )
        if tap:
            tap_a = events_q.filter(TapEvent.tap == "A").scalar() if tap == "A" else 0
            tap_b = events_q.filter(TapEvent.tap == "B").scalar() if tap == "B" else 0
        else:
            tap_a = events_q.filter(TapEvent.tap == "A").scalar()
            tap_b = events_q.filter(TapEvent.tap == "B").scalar()

        results.append(
            CountResult(
                video_id=video.id,
                original_name=video.original_name,
                upload_date=video.upload_date,
                tap_a=tap_a,
                tap_b=tap_b,
                total=tap_a + tap_b,
            )
        )
    return results


@router.get("/summary", response_model=CountSummary)
def get_summary(db: Session = Depends(get_db)):
    tap_a = (
        db.query(func.coalesce(func.sum(TapEvent.count), 0)).filter(TapEvent.tap == "A").scalar()
    )

    tap_b = (
        db.query(func.coalesce(func.sum(TapEvent.count), 0)).filter(TapEvent.tap == "B").scalar()
    )

    video_count = db.query(Video).filter(Video.status == "completed").count()

    return CountSummary(
        tap_a_total=tap_a,
        tap_b_total=tap_b,
        grand_total=tap_a + tap_b,
        video_count=video_count,
    )

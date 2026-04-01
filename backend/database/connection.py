"""Database engine, session factory, and initialisation."""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.config import DATABASE_URL, DB_DIR


engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite + threads
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    """Create the data/db/ directory and all tables."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    from backend.database.models import Base
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a session, closes on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

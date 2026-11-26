"""Database engine + session factory for prediction logging.

Single engine per process, instantiated lazily so the API can boot even when
``DATABASE_URL`` is unset (local dev without a database). Use
``init_engine()`` from the FastAPI lifespan to validate connectivity at
startup, and ``get_engine()`` from request handlers to grab the cached
instance.
"""

from __future__ import annotations

import logging
from threading import Lock

from sqlalchemy import Engine, create_engine

from api import settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None
_lock = Lock()


def init_engine(database_url: str | None = None) -> Engine | None:
    """Initialise the global engine. Returns None when no URL is configured.

    Called once at app startup. Safe to call multiple times — subsequent
    calls are no-ops.
    """
    global _engine
    url = database_url or settings.DATABASE_URL
    if not url:
        logger.warning(
            "DATABASE_URL not set — prediction logging disabled. "
            "Set the env var to enable Postgres persistence."
        )
        return None

    with _lock:
        if _engine is None:
            _engine = create_engine(
                url,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                future=True,
            )
            logger.info("Database engine initialised")
    return _engine


def get_engine() -> Engine | None:
    """Return the cached engine, or None if logging is disabled."""
    return _engine


def reset_engine() -> None:
    """Dispose and clear the engine. Useful for tests."""
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None

"""
Database configuration using aiosqlite (async SQLite).
Stores users and prediction history.
"""

import aiosqlite
import os

DB_PATH = os.getenv("DB_PATH", "cardioscan.db")


async def get_db():
    """Async context manager for DB connections."""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db


async def init_db():
    """Create tables if they don't exist."""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                username  TEXT    UNIQUE NOT NULL,
                email     TEXT    UNIQUE NOT NULL,
                password  TEXT    NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Separate schema preserves old history without presenting legacy scores
        # as results from the new methodology. Raw waveforms are never stored.
        await db.execute("""
            CREATE TABLE IF NOT EXISTS screening_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                case_id TEXT,
                model_version TEXT NOT NULL,
                result TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()

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

        await db.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id      INTEGER REFERENCES users(id),
                patient_name TEXT,
                age          INTEGER,
                sex          TEXT,
                signal_shape TEXT,
                predictions  TEXT,   -- JSON array of {class, prob, positive}
                top_class    TEXT,
                confidence   REAL,
                created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        await db.commit()

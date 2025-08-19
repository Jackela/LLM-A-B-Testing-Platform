"""Database configuration and session management."""

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

from sqlalchemy import MetaData, create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

# Create base class for all models
Base = declarative_base()

# Metadata with naming convention for constraints
naming_convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
Base.metadata = MetaData(naming_convention=naming_convention)


class DatabaseConfig:
    """Database configuration settings."""

    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        pool_size: int = 20,
        max_overflow: int = 30,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        pool_pre_ping: bool = True,
    ):
        self.database_url = database_url
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_timeout = pool_timeout
        self.pool_recycle = pool_recycle
        self.pool_pre_ping = pool_pre_ping

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create configuration from environment variables."""
        database_url = os.getenv(
            "DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/llm_ab_testing"
        )

        return cls(
            database_url=database_url,
            echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
            pool_size=int(os.getenv("DATABASE_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DATABASE_MAX_OVERFLOW", "30")),
            pool_timeout=int(os.getenv("DATABASE_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DATABASE_POOL_RECYCLE", "3600")),
            pool_pre_ping=os.getenv("DATABASE_POOL_PRE_PING", "true").lower() == "true",
        )


class DatabaseManager:
    """Database connection and session manager."""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._async_engine = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None

    def get_async_engine(self):
        """Get or create async database engine."""
        if self._async_engine is None:
            # Choose pool class based on database type
            if "sqlite" in self.config.database_url:
                poolclass = NullPool
                connect_args = {"check_same_thread": False}
            else:
                poolclass = QueuePool
                connect_args = {}

            # Build engine kwargs based on pool type
            engine_kwargs = {
                "echo": self.config.echo,
                "poolclass": poolclass,
                "pool_pre_ping": self.config.pool_pre_ping,
                "connect_args": connect_args,
                "future": True,
            }

            # Only add pool arguments for non-NullPool databases
            if poolclass != NullPool:
                engine_kwargs.update(
                    {
                        "pool_size": self.config.pool_size,
                        "max_overflow": self.config.max_overflow,
                        "pool_timeout": self.config.pool_timeout,
                        "pool_recycle": self.config.pool_recycle,
                    }
                )

            self._async_engine = create_async_engine(self.config.database_url, **engine_kwargs)
        return self._async_engine

    def get_sync_engine(self):
        """Get or create sync database engine (for migrations)."""
        if self._sync_engine is None:
            # Convert async URL to sync URL
            sync_url = self.config.database_url
            if "postgresql+asyncpg" in sync_url:
                sync_url = sync_url.replace("postgresql+asyncpg", "postgresql+psycopg2")
            elif "sqlite+aiosqlite" in sync_url:
                sync_url = sync_url.replace("sqlite+aiosqlite", "sqlite")

            if "sqlite" in sync_url:
                poolclass = NullPool
                connect_args = {"check_same_thread": False}
            else:
                poolclass = QueuePool
                connect_args = {}

            # Build sync engine kwargs based on pool type
            sync_engine_kwargs = {
                "echo": self.config.echo,
                "poolclass": poolclass,
                "pool_pre_ping": self.config.pool_pre_ping,
                "connect_args": connect_args,
                "future": True,
            }

            # Only add pool arguments for non-NullPool databases
            if poolclass != NullPool:
                sync_engine_kwargs.update(
                    {
                        "pool_size": self.config.pool_size,
                        "max_overflow": self.config.max_overflow,
                        "pool_timeout": self.config.pool_timeout,
                        "pool_recycle": self.config.pool_recycle,
                    }
                )

            self._sync_engine = create_engine(sync_url, **sync_engine_kwargs)
        return self._sync_engine

    def get_async_session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or create async session factory."""
        if self._async_session_factory is None:
            self._async_session_factory = async_sessionmaker(
                bind=self.get_async_engine(),
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
        return self._async_session_factory

    def get_sync_session_factory(self) -> sessionmaker:
        """Get or create sync session factory."""
        if self._sync_session_factory is None:
            self._sync_session_factory = sessionmaker(
                bind=self.get_sync_engine(),
                expire_on_commit=False,
                autoflush=True,
                autocommit=False,
            )
        return self._sync_session_factory

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session with automatic cleanup."""
        session_factory = self.get_async_session_factory()
        async with session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check database connection health."""
        try:
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1 as health_check"))
                row = result.fetchone()

                if row and row[0] == 1:
                    return {
                        "status": "healthy",
                        "database_url": self._mask_credentials(self.config.database_url),
                        "pool_info": self._get_pool_info(),
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": "Health check query failed",
                        "database_url": self._mask_credentials(self.config.database_url),
                    }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "database_url": self._mask_credentials(self.config.database_url),
            }

    def _mask_credentials(self, url: str) -> str:
        """Mask credentials in database URL for logging."""
        if "@" in url and "://" in url:
            protocol, rest = url.split("://", 1)
            if "@" in rest:
                credentials, server = rest.split("@", 1)
                return f"{protocol}://***:***@{server}"
        return url

    def _get_pool_info(self) -> Dict[str, Any]:
        """Get connection pool information."""
        engine = self.get_async_engine()
        pool = engine.pool

        if hasattr(pool, "size"):
            return {
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "total_connections": pool.size() + pool.overflow(),
            }
        else:
            return {"pool_type": "NullPool", "info": "No pooling"}

    async def close(self) -> None:
        """Close all database connections."""
        if self._async_engine:
            await self._async_engine.dispose()
        if self._sync_engine:
            self._sync_engine.dispose()


# Global database manager instance
_database_manager: Optional[DatabaseManager] = None


def init_database(config: Optional[DatabaseConfig] = None) -> DatabaseManager:
    """Initialize database manager."""
    global _database_manager

    if config is None:
        config = DatabaseConfig.from_env()

    _database_manager = DatabaseManager(config)
    return _database_manager


def get_database() -> DatabaseManager:
    """Get the current database manager."""
    if _database_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    return _database_manager


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Get database session from global manager."""
    db = get_database()
    async with db.get_session() as session:
        yield session


# Session factory type for dependency injection
SessionFactory = async_sessionmaker[AsyncSession]

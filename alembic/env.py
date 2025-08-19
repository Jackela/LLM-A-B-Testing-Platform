"""Alembic environment configuration."""

import asyncio
import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlalchemy.ext.asyncio import AsyncEngine

from alembic import context

# Import all models to ensure they are registered with metadata
from src.infrastructure.persistence.database import Base
from src.infrastructure.persistence.models import (
    test_models,
    provider_models,
    evaluation_models,
    analytics_models
)

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def get_database_url():
    """Get database URL from environment or config with validation."""
    database_url = os.getenv("DATABASE_URL")
    
    if database_url:
        print(f"[ALEMBIC] Using DATABASE_URL from environment")
        # Convert async URL to sync URL for Alembic
        if "postgresql+asyncpg" in database_url:
            database_url = database_url.replace("postgresql+asyncpg", "postgresql+psycopg2")
        elif "sqlite+aiosqlite" in database_url:
            database_url = database_url.replace("sqlite+aiosqlite", "sqlite")
        
        print(f"[ALEMBIC] Converted database URL: {database_url.split('://')[0]}://***")
        return database_url
    
    # Fallback to config file
    config_url = config.get_main_option("sqlalchemy.url")
    if config_url:
        print(f"[ALEMBIC] Using database URL from alembic.ini")
        return config_url
    
    # Default to PostgreSQL for CI environment
    default_url = "postgresql://postgres:postgres@localhost:5432/test_db"
    print(f"[ALEMBIC] Using default PostgreSQL URL for CI environment")
    return default_url


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    """Run migrations with connection."""
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations():
    """Run migrations in async mode."""
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = get_database_url()
    
    connectable = AsyncEngine(
        engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            future=True,
        )
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Check if we're running in async mode
    if os.getenv("ALEMBIC_ASYNC", "false").lower() == "true":
        asyncio.run(run_async_migrations())
    else:
        # Run in sync mode
        configuration = config.get_section(config.config_ini_section)
        configuration["sqlalchemy.url"] = get_database_url()
        
        connectable = engine_from_config(
            configuration,
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )

        with connectable.connect() as connection:
            do_run_migrations(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
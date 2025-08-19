"""Integration tests for database connection and performance."""

from datetime import datetime

import pytest
import pytest_asyncio

from src.infrastructure.persistence.connection_pool import ConnectionPoolManager
from src.infrastructure.persistence.database import DatabaseConfig, DatabaseManager
from src.infrastructure.persistence.query_optimizer import QueryOptimizer


@pytest_asyncio.fixture
async def database_manager():
    """Create test database manager."""
    config = DatabaseConfig(
        database_url="sqlite+aiosqlite:///:memory:", echo=False, pool_size=5, max_overflow=10
    )
    manager = DatabaseManager(config)

    # Create tables
    from src.infrastructure.persistence.database import Base

    async with manager.get_async_engine().begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield manager

    await manager.close()


@pytest_asyncio.fixture
async def pool_manager(database_manager):
    """Create connection pool manager."""
    return ConnectionPoolManager(database_manager)


@pytest_asyncio.fixture
async def query_optimizer():
    """Create query optimizer."""
    return QueryOptimizer()


class TestDatabaseConnection:
    """Test database connection and management."""

    async def test_database_health_check(self, database_manager):
        """Test database health check."""
        health = await database_manager.health_check()

        assert health["status"] == "healthy"
        assert "database_url" in health
        assert "pool_info" in health

    async def test_session_context_manager(self, database_manager):
        """Test session context manager."""
        async with database_manager.get_session() as session:
            # Execute simple query
            result = await session.execute("SELECT 1 as test")
            row = result.fetchone()
            assert row[0] == 1

    async def test_session_rollback_on_error(self, database_manager):
        """Test session rollback on error."""
        try:
            async with database_manager.get_session() as session:
                # Execute invalid query to trigger error
                await session.execute("SELECT * FROM non_existent_table")
        except Exception:
            # Exception should be propagated
            pass

        # Connection should still be healthy
        health = await database_manager.health_check()
        assert health["status"] == "healthy"

    async def test_multiple_concurrent_sessions(self, database_manager):
        """Test multiple concurrent sessions."""
        import asyncio

        async def execute_query(query_id):
            async with database_manager.get_session() as session:
                result = await session.execute(f"SELECT {query_id} as id")
                row = result.fetchone()
                return row[0]

        # Execute multiple queries concurrently
        tasks = [execute_query(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert results == [0, 1, 2, 3, 4]


class TestConnectionPoolManager:
    """Test connection pool management."""

    async def test_pool_metrics(self, pool_manager):
        """Test getting pool metrics."""
        metrics = await pool_manager.get_pool_metrics()

        assert metrics.pool_size >= 0
        assert metrics.checked_out >= 0
        assert metrics.checked_in >= 0
        assert metrics.last_updated is not None

    async def test_monitored_session(self, pool_manager):
        """Test monitored session execution."""
        async with pool_manager.get_monitored_session() as session:
            result = await session.execute("SELECT 1 as test")
            row = result.fetchone()
            assert row[0] == 1

    async def test_connection_health_check(self, pool_manager):
        """Test connection health check."""
        health = await pool_manager.health_check()

        assert isinstance(health.is_healthy, bool)
        assert isinstance(health.response_time_ms, float)
        assert health.last_check is not None

    async def test_connection_info(self, pool_manager):
        """Test getting connection information."""
        info = await pool_manager.get_connection_info()

        assert "database_url" in info
        assert "pool_metrics" in info
        assert "health_status" in info
        assert "engine_info" in info

    async def test_warm_up_pool(self, pool_manager):
        """Test warming up connection pool."""
        result = await pool_manager.warm_up_pool(target_connections=3)

        assert "target_connections" in result
        assert "successful_connections" in result
        assert "failed_connections" in result
        assert "success_rate" in result

    async def test_close_idle_connections(self, pool_manager):
        """Test closing idle connections."""
        result = await pool_manager.close_idle_connections()

        assert "connections_before" in result
        assert "connections_after" in result
        assert "connections_closed" in result


class TestQueryOptimizer:
    """Test query optimization utilities."""

    async def test_monitor_query(self, database_manager, query_optimizer):
        """Test query monitoring."""
        async with database_manager.get_session() as session:
            async with query_optimizer.monitor_query(session, "test_query"):
                result = await session.execute("SELECT 1 as test")
                row = result.fetchone()
                assert row[0] == 1

    async def test_query_performance_summary(self, query_optimizer):
        """Test query performance summary."""
        summary = await query_optimizer.get_query_performance_summary()

        assert "total_queries" in summary
        assert "average_execution_time" in summary
        assert "slow_queries" in summary

    async def test_table_statistics(self, database_manager, query_optimizer):
        """Test getting table statistics."""
        async with database_manager.get_session() as session:
            # This might not work with SQLite, but test the structure
            stats = await query_optimizer.get_table_statistics(session, "tests")

            assert "table_name" in stats
            assert "column_statistics" in stats
            assert "recommendations" in stats

    async def test_index_usage_statistics(self, database_manager, query_optimizer):
        """Test getting index usage statistics."""
        async with database_manager.get_session() as session:
            stats = await query_optimizer.get_index_usage_statistics(session)

            assert "index_statistics" in stats
            assert "unused_indexes" in stats
            assert "recommendations" in stats

    async def test_optimize_table_indexes(self, database_manager, query_optimizer):
        """Test table index optimization."""
        async with database_manager.get_session() as session:
            optimization = await query_optimizer.optimize_table_indexes(session, "tests")

            assert "table_name" in optimization
            assert "current_indexes" in optimization
            assert "query_patterns" in optimization
            assert "recommendations" in optimization


class TestDatabaseInitialization:
    """Test database initialization and configuration."""

    def test_database_config_from_env(self, monkeypatch):
        """Test database configuration from environment."""
        # Set environment variables
        monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@localhost/test")
        monkeypatch.setenv("DATABASE_POOL_SIZE", "15")
        monkeypatch.setenv("DATABASE_ECHO", "true")

        config = DatabaseConfig.from_env()

        assert config.database_url == "postgresql://test:test@localhost/test"
        assert config.pool_size == 15
        assert config.echo is True

    def test_database_config_defaults(self):
        """Test database configuration defaults."""
        config = DatabaseConfig.from_env()

        assert config.pool_size == 20
        assert config.max_overflow == 30
        assert config.pool_timeout == 30
        assert config.echo is False

    async def test_database_manager_creation(self):
        """Test database manager creation."""
        config = DatabaseConfig(database_url="sqlite+aiosqlite:///:memory:", pool_size=5)

        manager = DatabaseManager(config)

        assert manager.config == config
        assert manager._async_engine is None  # Not created until needed

        # Create engine
        engine = manager.get_async_engine()
        assert engine is not None
        assert manager._async_engine is engine  # Cached

        # Create session factory
        session_factory = manager.get_async_session_factory()
        assert session_factory is not None

        await manager.close()

    async def test_credential_masking(self):
        """Test credential masking in URLs."""
        config = DatabaseConfig(database_url="postgresql://user:password@localhost:5432/test")
        manager = DatabaseManager(config)

        masked_url = manager._mask_credentials(config.database_url)
        assert "***:***" in masked_url
        assert "password" not in masked_url
        assert "user" not in masked_url

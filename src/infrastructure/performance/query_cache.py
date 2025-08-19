"""Advanced query result caching with intelligent invalidation."""

import asyncio
import hashlib
import time
import weakref
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Set, Tuple, Union

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Delete, Insert, Select, Update

from .cache_manager import CacheLayer, CacheManager


@dataclass
class QueryCacheEntry:
    """Query cache entry with metadata."""

    key: str
    query_hash: str
    result: Any
    tables_accessed: Set[str]
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    ttl_seconds: Optional[int] = None
    invalidation_rules: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False

        return (datetime.utcnow() - self.created_at).total_seconds() > self.ttl_seconds

    def touch(self) -> None:
        """Update last accessed time and increment access count."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class QueryInvalidationRule:
    """Rule for determining when to invalidate query cache entries."""

    def __init__(
        self,
        name: str,
        table_patterns: List[str],
        operation_types: List[str] = None,
        custom_condition: Optional[Callable] = None,
    ):
        self.name = name
        self.table_patterns = table_patterns
        self.operation_types = operation_types or ["INSERT", "UPDATE", "DELETE"]
        self.custom_condition = custom_condition

    def should_invalidate(
        self, operation_type: str, table_name: str, operation_data: Optional[Dict] = None
    ) -> bool:
        """Check if this rule should trigger cache invalidation."""
        # Check operation type
        if operation_type.upper() not in self.operation_types:
            return False

        # Check table patterns
        table_match = any(
            self._matches_pattern(table_name, pattern) for pattern in self.table_patterns
        )

        if not table_match:
            return False

        # Check custom condition if provided
        if self.custom_condition:
            return self.custom_condition(operation_type, table_name, operation_data)

        return True

    def _matches_pattern(self, table_name: str, pattern: str) -> bool:
        """Check if table name matches pattern (supports wildcards)."""
        import fnmatch

        return fnmatch.fnmatch(table_name.lower(), pattern.lower())


class QueryCache:
    """Advanced query result cache with intelligent invalidation."""

    def __init__(
        self,
        cache_manager: CacheManager,
        default_ttl: int = 300,  # 5 minutes
        max_entries: int = 10000,
        enable_query_analysis: bool = True,
    ):
        self.cache_manager = cache_manager
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self.enable_query_analysis = enable_query_analysis

        # Cache metadata
        self._cache_entries: Dict[str, QueryCacheEntry] = {}
        self._table_to_queries: Dict[str, Set[str]] = defaultdict(set)
        self._invalidation_rules: List[QueryInvalidationRule] = []

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "invalidations": 0,
            "evictions": 0,
            "total_queries_cached": 0,
        }

        # Setup default invalidation rules
        self._setup_default_invalidation_rules()

    def _setup_default_invalidation_rules(self) -> None:
        """Setup default cache invalidation rules."""
        # Invalidate all caches when test configurations change
        self.add_invalidation_rule(
            QueryInvalidationRule(
                name="test_configuration_changes",
                table_patterns=["test_configurations", "tests"],
                operation_types=["INSERT", "UPDATE", "DELETE"],
            )
        )

        # Invalidate related caches when test samples are added
        self.add_invalidation_rule(
            QueryInvalidationRule(
                name="test_sample_changes",
                table_patterns=["test_samples"],
                operation_types=["INSERT", "UPDATE", "DELETE"],
            )
        )

        # Invalidate analytics caches when new results are added
        self.add_invalidation_rule(
            QueryInvalidationRule(
                name="analytics_changes",
                table_patterns=["analytics_*", "results_*"],
                operation_types=["INSERT", "UPDATE", "DELETE"],
            )
        )

        # Invalidate provider caches when provider data changes
        self.add_invalidation_rule(
            QueryInvalidationRule(
                name="provider_changes",
                table_patterns=["model_providers", "provider_*"],
                operation_types=["INSERT", "UPDATE", "DELETE"],
            )
        )

    def add_invalidation_rule(self, rule: QueryInvalidationRule) -> None:
        """Add a cache invalidation rule."""
        self._invalidation_rules.append(rule)

    def _generate_query_key(
        self, query: str, params: Optional[Dict] = None, namespace: str = "query_cache"
    ) -> str:
        """Generate cache key for query."""
        # Normalize query (remove extra whitespace, convert to lowercase)
        normalized_query = " ".join(query.lower().split())

        # Include parameters in key
        param_str = ""
        if params:
            sorted_params = sorted(params.items())
            param_str = "_".join(f"{k}={v}" for k, v in sorted_params)

        # Generate hash
        key_data = f"{normalized_query}_{param_str}"
        query_hash = hashlib.sha256(key_data.encode()).hexdigest()

        return f"{namespace}:query:{query_hash[:16]}"

    def _extract_tables_from_query(self, query: str) -> Set[str]:
        """Extract table names from SQL query (simplified approach)."""
        tables = set()

        # This is a simplified table extraction - in production you'd use a proper SQL parser
        query_lower = query.lower()

        # Common patterns for table references
        keywords = ["from", "join", "update", "insert into", "delete from"]

        for keyword in keywords:
            if keyword in query_lower:
                # Find table name after keyword
                parts = query_lower.split(keyword)
                if len(parts) > 1:
                    # Get the next word after the keyword
                    next_part = parts[1].strip()
                    table_name = next_part.split()[0] if next_part else ""

                    # Clean up table name (remove schema prefix, quotes, etc.)
                    table_name = table_name.replace('"', "").replace("'", "")
                    if "." in table_name:
                        table_name = table_name.split(".")[-1]  # Remove schema

                    if table_name and table_name not in ["(", "select", "where", "order", "group"]:
                        tables.add(table_name)

        return tables

    async def get_cached_query_result(
        self, query: str, params: Optional[Dict] = None, namespace: str = "query_cache"
    ) -> Optional[Any]:
        """Get cached query result."""
        cache_key = self._generate_query_key(query, params, namespace)

        # Check if we have metadata for this query
        if cache_key in self._cache_entries:
            entry = self._cache_entries[cache_key]

            # Check if expired
            if entry.is_expired():
                await self._remove_cache_entry(cache_key)
                self._stats["misses"] += 1
                return None

            # Get from cache
            result = await self.cache_manager.get(cache_key, namespace)
            if result is not None:
                entry.touch()
                self._stats["hits"] += 1
                return result
            else:
                # Cache miss - remove stale metadata
                await self._remove_cache_entry(cache_key)

        self._stats["misses"] += 1
        return None

    async def cache_query_result(
        self,
        query: str,
        result: Any,
        params: Optional[Dict] = None,
        ttl: Optional[int] = None,
        namespace: str = "query_cache",
        tables_accessed: Optional[Set[str]] = None,
    ) -> bool:
        """Cache query result with metadata."""
        cache_key = self._generate_query_key(query, params, namespace)
        ttl = ttl or self.default_ttl

        # Extract tables if not provided
        if tables_accessed is None and self.enable_query_analysis:
            tables_accessed = self._extract_tables_from_query(query)
        elif tables_accessed is None:
            tables_accessed = set()

        # Check cache size and evict if necessary
        await self._evict_if_necessary()

        # Cache the result
        success = await self.cache_manager.set(cache_key, result, ttl=ttl, namespace=namespace)

        if success:
            # Store metadata
            query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
            entry = QueryCacheEntry(
                key=cache_key,
                query_hash=query_hash,
                result=result,
                tables_accessed=tables_accessed,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl_seconds=ttl,
            )

            self._cache_entries[cache_key] = entry

            # Update table to query mapping
            for table in tables_accessed:
                self._table_to_queries[table].add(cache_key)

            self._stats["total_queries_cached"] += 1

        return success

    @asynccontextmanager
    async def cached_query(
        self,
        session: AsyncSession,
        query: Union[str, Select],
        params: Optional[Dict] = None,
        ttl: Optional[int] = None,
        namespace: str = "query_cache",
        tables_accessed: Optional[Set[str]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Context manager for cached query execution."""
        # Convert SQLAlchemy query to string if necessary
        if hasattr(query, "compile"):
            query_str = str(query.compile(compile_kwargs={"literal_binds": True}))
        else:
            query_str = str(query)

        # Try cache first
        cached_result = await self.get_cached_query_result(query_str, params, namespace)
        if cached_result is not None:
            yield cached_result
            return

        # Execute query and cache result
        if hasattr(query, "compile"):
            # SQLAlchemy query object
            if params:
                result = await session.execute(query, params)
            else:
                result = await session.execute(query)
        else:
            # Raw SQL string
            if params:
                result = await session.execute(text(query_str), params)
            else:
                result = await session.execute(text(query_str))

        # Fetch results (assuming we want all rows)
        if hasattr(result, "fetchall"):
            query_result = result.fetchall()
        else:
            query_result = result

        # Cache the result
        await self.cache_query_result(
            query_str, query_result, params, ttl, namespace, tables_accessed
        )

        yield query_result

    async def invalidate_by_table(self, table_name: str) -> int:
        """Invalidate all cached queries that access the specified table."""
        invalidated_count = 0

        if table_name in self._table_to_queries:
            cache_keys = list(self._table_to_queries[table_name])

            for cache_key in cache_keys:
                await self._remove_cache_entry(cache_key)
                invalidated_count += 1

        self._stats["invalidations"] += invalidated_count
        return invalidated_count

    async def invalidate_by_pattern(self, table_pattern: str) -> int:
        """Invalidate cached queries that access tables matching the pattern."""
        import fnmatch

        invalidated_count = 0
        tables_to_invalidate = [
            table
            for table in self._table_to_queries.keys()
            if fnmatch.fnmatch(table.lower(), table_pattern.lower())
        ]

        for table in tables_to_invalidate:
            count = await self.invalidate_by_table(table)
            invalidated_count += count

        return invalidated_count

    async def handle_data_change(
        self, operation_type: str, table_name: str, operation_data: Optional[Dict] = None
    ) -> int:
        """Handle data change and invalidate relevant caches."""
        invalidated_count = 0

        for rule in self._invalidation_rules:
            if rule.should_invalidate(operation_type, table_name, operation_data):
                # Apply rule-specific invalidation
                for pattern in rule.table_patterns:
                    count = await self.invalidate_by_pattern(pattern)
                    invalidated_count += count

        return invalidated_count

    async def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove cache entry and its metadata."""
        # Remove from cache
        await self.cache_manager.delete(cache_key)

        # Remove metadata
        if cache_key in self._cache_entries:
            entry = self._cache_entries[cache_key]

            # Remove from table mappings
            for table in entry.tables_accessed:
                if table in self._table_to_queries:
                    self._table_to_queries[table].discard(cache_key)

                    # Clean up empty table entries
                    if not self._table_to_queries[table]:
                        del self._table_to_queries[table]

            del self._cache_entries[cache_key]

    async def _evict_if_necessary(self) -> int:
        """Evict least recently used entries if cache is full."""
        if len(self._cache_entries) < self.max_entries:
            return 0

        # Sort by last accessed time
        sorted_entries = sorted(self._cache_entries.items(), key=lambda x: x[1].last_accessed)

        # Evict oldest 10% of entries
        evict_count = max(1, int(self.max_entries * 0.1))
        evicted = 0

        for cache_key, _ in sorted_entries[:evict_count]:
            await self._remove_cache_entry(cache_key)
            evicted += 1

        self._stats["evictions"] += evicted
        return evicted

    async def clear_cache(self, namespace: str = "query_cache") -> None:
        """Clear all cached queries."""
        await self.cache_manager.clear(namespace)

        # Clear metadata
        self._cache_entries.clear()
        self._table_to_queries.clear()

    async def warm_cache(
        self, session: AsyncSession, warm_queries: List[Tuple[str, Optional[Dict], Optional[int]]]
    ) -> Dict[str, Any]:
        """Warm up cache with frequently used queries."""
        warmed_count = 0
        failed_count = 0

        for query, params, ttl in warm_queries:
            try:
                async with self.cached_query(session, query, params, ttl) as result:
                    # Query executed and cached
                    warmed_count += 1
            except Exception as e:
                print(f"Failed to warm cache for query: {e}")
                failed_count += 1

        return {
            "warmed_queries": warmed_count,
            "failed_queries": failed_count,
            "total_cached_queries": len(self._cache_entries),
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "cache_hits": self._stats["hits"],
            "cache_misses": self._stats["misses"],
            "invalidations": self._stats["invalidations"],
            "evictions": self._stats["evictions"],
            "total_queries_cached": self._stats["total_queries_cached"],
            "current_cache_size": len(self._cache_entries),
            "max_cache_size": self.max_entries,
            "cache_utilization": len(self._cache_entries) / self.max_entries,
            "tables_tracked": len(self._table_to_queries),
            "invalidation_rules": len(self._invalidation_rules),
        }

    def get_top_cached_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top cached queries by access count."""
        sorted_entries = sorted(
            self._cache_entries.values(), key=lambda x: x.access_count, reverse=True
        )

        return [
            {
                "query_hash": entry.query_hash,
                "access_count": entry.access_count,
                "tables_accessed": list(entry.tables_accessed),
                "created_at": entry.created_at.isoformat(),
                "last_accessed": entry.last_accessed.isoformat(),
                "is_expired": entry.is_expired(),
            }
            for entry in sorted_entries[:limit]
        ]


class QueryCacheManager:
    """Manager for multiple query caches with different configurations."""

    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self._caches: Dict[str, QueryCache] = {}

    def get_cache(
        self,
        name: str,
        default_ttl: int = 300,
        max_entries: int = 10000,
        enable_query_analysis: bool = True,
    ) -> QueryCache:
        """Get or create a named query cache."""
        if name not in self._caches:
            self._caches[name] = QueryCache(
                self.cache_manager, default_ttl, max_entries, enable_query_analysis
            )

        return self._caches[name]

    def get_analytics_cache(self) -> QueryCache:
        """Get cache optimized for analytics queries."""
        return self.get_cache(
            "analytics", default_ttl=600, max_entries=5000, enable_query_analysis=True  # 10 minutes
        )

    def get_test_data_cache(self) -> QueryCache:
        """Get cache for test data queries."""
        return self.get_cache(
            "test_data", default_ttl=300, max_entries=10000, enable_query_analysis=True  # 5 minutes
        )

    def get_provider_cache(self) -> QueryCache:
        """Get cache for provider data queries."""
        return self.get_cache(
            "provider_data",
            default_ttl=1800,  # 30 minutes
            max_entries=2000,
            enable_query_analysis=True,
        )

    async def invalidate_all_caches(self) -> Dict[str, int]:
        """Invalidate all caches."""
        results = {}

        for name, cache in self._caches.items():
            await cache.clear_cache(namespace=f"query_cache_{name}")
            results[name] = "cleared"

        return results

    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        stats = {}

        for name, cache in self._caches.items():
            stats[name] = cache.get_cache_stats()

        return stats

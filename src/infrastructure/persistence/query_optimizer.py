"""Query optimization utilities and performance monitoring."""

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.engine import Result
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass
class QueryMetrics:
    """Query performance metrics."""

    query_id: str
    sql_statement: str
    execution_time_ms: float
    rows_returned: int
    rows_examined: int
    index_usage: List[str]
    warnings: List[str]
    executed_at: datetime


class QueryOptimizer:
    """Query optimization and performance monitoring."""

    def __init__(self):
        self._query_cache: Dict[str, QueryMetrics] = {}
        self._slow_query_threshold_ms = 1000  # 1 second
        self._max_cache_entries = 1000

    @asynccontextmanager
    async def monitor_query(
        self, session: AsyncSession, query_id: str = None
    ) -> AsyncGenerator[AsyncSession, None]:
        """Monitor query execution with performance metrics."""
        if query_id is None:
            query_id = f"query_{int(time.time() * 1000)}"

        start_time = time.time()

        try:
            yield session
        finally:
            execution_time = (time.time() - start_time) * 1000  # Convert to ms

            # Log slow queries
            if execution_time > self._slow_query_threshold_ms:
                await self._log_slow_query(session, query_id, execution_time)

    async def analyze_query_plan(self, session: AsyncSession, query: str) -> Dict[str, Any]:
        """Analyze query execution plan."""
        explain_query = f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}"

        try:
            result = await session.execute(text(explain_query))
            plan_data = result.fetchone()[0]

            return {
                "execution_plan": plan_data,
                "analysis": self._analyze_plan_data(plan_data),
                "recommendations": self._generate_plan_recommendations(plan_data),
            }
        except Exception as e:
            return {"error": str(e), "analysis": {}, "recommendations": []}

    async def get_table_statistics(self, session: AsyncSession, table_name: str) -> Dict[str, Any]:
        """Get table statistics for optimization."""
        stats_query = text(
            """
            SELECT 
                schemaname,
                tablename,
                attname,
                n_distinct,
                correlation,
                most_common_vals,
                most_common_freqs,
                histogram_bounds
            FROM pg_stats 
            WHERE tablename = :table_name
        """
        )

        try:
            result = await session.execute(stats_query, {"table_name": table_name})
            rows = result.fetchall()

            return {
                "table_name": table_name,
                "column_statistics": [
                    {
                        "column_name": row.attname,
                        "distinct_values": row.n_distinct,
                        "correlation": row.correlation,
                        "most_common_values": row.most_common_vals,
                        "most_common_frequencies": row.most_common_freqs,
                        "histogram_bounds": row.histogram_bounds,
                    }
                    for row in rows
                ],
                "recommendations": self._generate_table_recommendations(rows),
            }
        except Exception as e:
            return {
                "table_name": table_name,
                "error": str(e),
                "column_statistics": [],
                "recommendations": [],
            }

    async def get_index_usage_statistics(self, session: AsyncSession) -> Dict[str, Any]:
        """Get index usage statistics."""
        index_stats_query = text(
            """
            SELECT 
                schemaname,
                tablename,
                indexname,
                idx_tup_read,
                idx_tup_fetch,
                idx_scan,
                pg_size_pretty(pg_relation_size(indexrelid)) as index_size
            FROM pg_stat_user_indexes
            ORDER BY idx_scan DESC
        """
        )

        try:
            result = await session.execute(index_stats_query)
            rows = result.fetchall()

            return {
                "index_statistics": [
                    {
                        "schema": row.schemaname,
                        "table": row.tablename,
                        "index": row.indexname,
                        "tuples_read": row.idx_tup_read,
                        "tuples_fetched": row.idx_tup_fetch,
                        "scan_count": row.idx_scan,
                        "size": row.index_size,
                    }
                    for row in rows
                ],
                "unused_indexes": [row.indexname for row in rows if row.idx_scan == 0],
                "recommendations": self._generate_index_recommendations(rows),
            }
        except Exception as e:
            return {
                "error": str(e),
                "index_statistics": [],
                "unused_indexes": [],
                "recommendations": [],
            }

    async def get_query_performance_summary(self) -> Dict[str, Any]:
        """Get summary of query performance metrics."""
        if not self._query_cache:
            return {
                "total_queries": 0,
                "average_execution_time": 0,
                "slow_queries": 0,
                "performance_trends": [],
            }

        metrics = list(self._query_cache.values())
        execution_times = [m.execution_time_ms for m in metrics]

        return {
            "total_queries": len(metrics),
            "average_execution_time": sum(execution_times) / len(execution_times),
            "min_execution_time": min(execution_times),
            "max_execution_time": max(execution_times),
            "slow_queries": len([t for t in execution_times if t > self._slow_query_threshold_ms]),
            "slow_query_threshold": self._slow_query_threshold_ms,
            "most_common_warnings": self._get_common_warnings(metrics),
            "index_usage_patterns": self._analyze_index_usage_patterns(metrics),
        }

    async def optimize_table_indexes(
        self, session: AsyncSession, table_name: str
    ) -> Dict[str, Any]:
        """Analyze and suggest index optimizations for a table."""
        # Get current indexes
        current_indexes_query = text(
            """
            SELECT 
                indexname,
                indexdef
            FROM pg_indexes 
            WHERE tablename = :table_name
        """
        )

        # Get query patterns
        query_patterns_query = text(
            """
            SELECT 
                query,
                calls,
                mean_exec_time,
                rows
            FROM pg_stat_statements 
            WHERE query LIKE '%' || :table_name || '%'
            ORDER BY calls DESC
            LIMIT 20
        """
        )

        try:
            # Get current indexes
            indexes_result = await session.execute(
                current_indexes_query, {"table_name": table_name}
            )
            current_indexes = [
                {"name": row.indexname, "definition": row.indexdef}
                for row in indexes_result.fetchall()
            ]

            # Get query patterns (if pg_stat_statements is available)
            try:
                patterns_result = await session.execute(
                    query_patterns_query, {"table_name": table_name}
                )
                query_patterns = [
                    {
                        "query": row.query,
                        "calls": row.calls,
                        "avg_time": row.mean_exec_time,
                        "avg_rows": row.rows,
                    }
                    for row in patterns_result.fetchall()
                ]
            except:
                query_patterns = []

            # Generate recommendations
            recommendations = self._generate_index_optimization_recommendations(
                table_name, current_indexes, query_patterns
            )

            return {
                "table_name": table_name,
                "current_indexes": current_indexes,
                "query_patterns": query_patterns,
                "recommendations": recommendations,
            }

        except Exception as e:
            return {
                "table_name": table_name,
                "error": str(e),
                "current_indexes": [],
                "query_patterns": [],
                "recommendations": [],
            }

    async def _log_slow_query(
        self, session: AsyncSession, query_id: str, execution_time: float
    ) -> None:
        """Log slow query for analysis."""
        # In a real implementation, this would log to a monitoring system
        print(f"SLOW QUERY DETECTED: {query_id} took {execution_time:.2f}ms")

    def _analyze_plan_data(self, plan_data: List[Dict]) -> Dict[str, Any]:
        """Analyze query execution plan data."""
        if not plan_data:
            return {}

        plan = plan_data[0]["Plan"]

        analysis = {
            "total_cost": plan.get("Total Cost", 0),
            "actual_time": plan.get("Actual Total Time", 0),
            "rows_returned": plan.get("Actual Rows", 0),
            "node_type": plan.get("Node Type"),
            "scan_methods": self._extract_scan_methods(plan),
            "join_methods": self._extract_join_methods(plan),
            "sort_operations": self._extract_sort_operations(plan),
            "index_usage": self._extract_index_usage(plan),
        }

        return analysis

    def _generate_plan_recommendations(self, plan_data: List[Dict]) -> List[str]:
        """Generate recommendations based on execution plan."""
        if not plan_data:
            return []

        recommendations = []
        plan = plan_data[0]["Plan"]

        # Check for sequential scans
        if self._has_sequential_scans(plan):
            recommendations.append("Consider adding indexes for sequential scans")

        # Check for sort operations
        if self._has_expensive_sorts(plan):
            recommendations.append("Consider adding indexes to avoid sorting")

        # Check for nested loops
        if self._has_nested_loops(plan):
            recommendations.append("Consider optimizing join conditions or adding indexes")

        return recommendations

    def _generate_table_recommendations(self, stats_rows: List) -> List[str]:
        """Generate recommendations based on table statistics."""
        recommendations = []

        for row in stats_rows:
            if row.n_distinct and row.n_distinct < -0.1:  # High cardinality
                recommendations.append(f"Consider indexing high-cardinality column: {row.attname}")

            if row.correlation and abs(row.correlation) > 0.8:
                recommendations.append(
                    f"Column {row.attname} has high correlation - good for range queries"
                )

        return recommendations

    def _generate_index_recommendations(self, index_rows: List) -> List[str]:
        """Generate index recommendations."""
        recommendations = []

        unused_indexes = [row for row in index_rows if row.idx_scan == 0]
        if unused_indexes:
            recommendations.append(f"Consider dropping {len(unused_indexes)} unused indexes")

        low_usage_indexes = [row for row in index_rows if 0 < row.idx_scan < 10]
        if low_usage_indexes:
            recommendations.append(f"Review {len(low_usage_indexes)} low-usage indexes")

        return recommendations

    def _generate_index_optimization_recommendations(
        self, table_name: str, current_indexes: List[Dict], query_patterns: List[Dict]
    ) -> List[str]:
        """Generate specific index optimization recommendations."""
        recommendations = []

        if len(current_indexes) > 10:
            recommendations.append("High number of indexes - consider consolidating")

        if query_patterns:
            # Analyze query patterns for common WHERE clauses
            common_columns = self._extract_common_where_columns(query_patterns)
            for column in common_columns:
                if not self._has_index_on_column(current_indexes, column):
                    recommendations.append(
                        f"Consider adding index on frequently queried column: {column}"
                    )

        return recommendations

    def _extract_scan_methods(self, plan: Dict) -> List[str]:
        """Extract scan methods from execution plan."""
        methods = []

        def extract_recursive(node):
            if "Node Type" in node:
                if "Scan" in node["Node Type"]:
                    methods.append(node["Node Type"])

            if "Plans" in node:
                for child in node["Plans"]:
                    extract_recursive(child)

        extract_recursive(plan)
        return methods

    def _extract_join_methods(self, plan: Dict) -> List[str]:
        """Extract join methods from execution plan."""
        methods = []

        def extract_recursive(node):
            if "Node Type" in node:
                if "Join" in node["Node Type"]:
                    methods.append(node["Node Type"])

            if "Plans" in node:
                for child in node["Plans"]:
                    extract_recursive(child)

        extract_recursive(plan)
        return methods

    def _extract_sort_operations(self, plan: Dict) -> List[Dict]:
        """Extract sort operations from execution plan."""
        sorts = []

        def extract_recursive(node):
            if "Node Type" in node and node["Node Type"] == "Sort":
                sorts.append(
                    {
                        "sort_keys": node.get("Sort Key", []),
                        "sort_method": node.get("Sort Method"),
                        "memory_used": node.get("Sort Space Used"),
                    }
                )

            if "Plans" in node:
                for child in node["Plans"]:
                    extract_recursive(child)

        extract_recursive(plan)
        return sorts

    def _extract_index_usage(self, plan: Dict) -> List[str]:
        """Extract index usage from execution plan."""
        indexes = []

        def extract_recursive(node):
            if "Index Name" in node:
                indexes.append(node["Index Name"])

            if "Plans" in node:
                for child in node["Plans"]:
                    extract_recursive(child)

        extract_recursive(plan)
        return indexes

    def _has_sequential_scans(self, plan: Dict) -> bool:
        """Check if plan contains sequential scans."""
        return "Seq Scan" in str(plan)

    def _has_expensive_sorts(self, plan: Dict) -> bool:
        """Check if plan contains expensive sort operations."""
        # This is a simplified check
        return "Sort" in str(plan) and "external" in str(plan).lower()

    def _has_nested_loops(self, plan: Dict) -> bool:
        """Check if plan contains nested loop joins."""
        return "Nested Loop" in str(plan)

    def _get_common_warnings(self, metrics: List[QueryMetrics]) -> List[str]:
        """Get most common warnings from query metrics."""
        warning_counts = {}
        for metric in metrics:
            for warning in metric.warnings:
                warning_counts[warning] = warning_counts.get(warning, 0) + 1

        return sorted(warning_counts.keys(), key=warning_counts.get, reverse=True)[:5]

    def _analyze_index_usage_patterns(self, metrics: List[QueryMetrics]) -> Dict[str, int]:
        """Analyze index usage patterns."""
        index_usage = {}
        for metric in metrics:
            for index in metric.index_usage:
                index_usage[index] = index_usage.get(index, 0) + 1

        return dict(sorted(index_usage.items(), key=lambda x: x[1], reverse=True)[:10])

    def _extract_common_where_columns(self, query_patterns: List[Dict]) -> List[str]:
        """Extract commonly used columns in WHERE clauses."""
        # This is a simplified implementation
        # In reality, you'd parse SQL to extract column names
        common_columns = []
        for pattern in query_patterns:
            query = pattern["query"].lower()
            if "where" in query:
                # Simple heuristic - look for common column patterns
                if "id =" in query:
                    common_columns.append("id")
                if "status =" in query:
                    common_columns.append("status")
                if "created_at" in query:
                    common_columns.append("created_at")

        return list(set(common_columns))

    def _has_index_on_column(self, indexes: List[Dict], column: str) -> bool:
        """Check if there's an index on the specified column."""
        for index in indexes:
            if column in index["definition"].lower():
                return True
        return False

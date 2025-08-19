"""API response optimization with compression, caching, and batching."""

import asyncio
import gzip
import hashlib
import json
import time
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Union

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .cache_manager import CacheLayer, CacheManager
from .metrics_collector import MetricsCollector


class CompressionType(Enum):
    """Compression algorithm types."""

    NONE = "none"
    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"


@dataclass
class ResponseMetrics:
    """API response performance metrics."""

    total_responses: int = 0
    compressed_responses: int = 0
    cached_responses: int = 0
    average_response_size_bytes: float = 0.0
    average_compression_ratio: float = 0.0
    compression_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    batched_requests: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ResponseCompressor:
    """Advanced response compression with algorithm selection."""

    def __init__(self, min_size: int = 1024, compression_level: int = 6):
        self.min_size = min_size
        self.compression_level = compression_level

        # Algorithm preferences (in order of preference)
        self._algorithm_preferences = [
            CompressionType.GZIP,
            CompressionType.DEFLATE,
            CompressionType.NONE,
        ]

    def select_compression_algorithm(
        self, accept_encoding: str, content_type: str, content_size: int
    ) -> CompressionType:
        """Select optimal compression algorithm based on client support and content."""
        # Don't compress small responses
        if content_size < self.min_size:
            return CompressionType.NONE

        # Don't compress already compressed content
        if any(
            ct in content_type.lower() for ct in ["image/", "video/", "audio/", "application/zip"]
        ):
            return CompressionType.NONE

        # Parse Accept-Encoding header
        accepted_encodings = [enc.strip().lower() for enc in accept_encoding.split(",")]

        # Check for supported algorithms in preference order
        for algorithm in self._algorithm_preferences:
            if algorithm == CompressionType.NONE:
                return algorithm

            if algorithm.value in accepted_encodings:
                return algorithm

        return CompressionType.NONE

    def compress_response(self, content: bytes, algorithm: CompressionType) -> bytes:
        """Compress response content using specified algorithm."""
        if algorithm == CompressionType.NONE:
            return content

        start_time = time.time()

        try:
            if algorithm == CompressionType.GZIP:
                compressed = gzip.compress(content, compresslevel=self.compression_level)
            elif algorithm == CompressionType.DEFLATE:
                compressed = zlib.compress(content, level=self.compression_level)
            else:
                # Fallback to no compression
                compressed = content

            compression_time = (time.time() - start_time) * 1000
            compression_ratio = len(content) / len(compressed) if compressed else 1.0

            return compressed, compression_time, compression_ratio

        except Exception as e:
            print(f"Compression failed: {e}")
            return content, 0, 1.0

    def get_content_encoding(self, algorithm: CompressionType) -> Optional[str]:
        """Get Content-Encoding header value for algorithm."""
        if algorithm == CompressionType.GZIP:
            return "gzip"
        elif algorithm == CompressionType.DEFLATE:
            return "deflate"
        else:
            return None


class ResponseCache:
    """Advanced response caching with intelligent invalidation."""

    def __init__(self, cache_manager: CacheManager, default_ttl: int = 300):
        self.cache_manager = cache_manager
        self.default_ttl = default_ttl
        self._cache_key_patterns: Dict[str, Callable] = {}

    def register_cache_pattern(self, pattern: str, key_generator: Callable) -> None:
        """Register a cache key generation pattern for specific endpoints."""
        self._cache_key_patterns[pattern] = key_generator

    def generate_cache_key(
        self, request: Request, additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key for request."""
        # Base key components
        method = request.method
        path = str(request.url.path)
        query_params = dict(request.query_params)

        # Check for custom key generator
        for pattern, generator in self._cache_key_patterns.items():
            if pattern in path:
                return generator(request, additional_context)

        # Default key generation
        key_components = [method, path, json.dumps(query_params, sort_keys=True)]

        if additional_context:
            key_components.append(json.dumps(additional_context, sort_keys=True))

        key_string = "|".join(key_components)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    async def get_cached_response(
        self, request: Request, additional_context: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        cache_key = self.generate_cache_key(request, additional_context)

        cached_data = await self.cache_manager.get(
            cache_key, namespace="api_responses", layer=CacheLayer.HYBRID
        )

        return cached_data

    async def cache_response(
        self,
        request: Request,
        response_data: Any,
        ttl: Optional[int] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Cache response data."""
        cache_key = self.generate_cache_key(request, additional_context)
        ttl = ttl or self.default_ttl

        # Prepare cache data
        cache_data = {"data": response_data, "cached_at": datetime.utcnow().isoformat(), "ttl": ttl}

        return await self.cache_manager.set(
            cache_key, cache_data, ttl=ttl, namespace="api_responses", layer=CacheLayer.HYBRID
        )

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cached responses matching pattern."""
        # This would typically be implemented by the cache manager
        # For now, we'll clear the entire namespace
        await self.cache_manager.clear("api_responses")
        return 1  # Simplified return


class RequestBatcher:
    """Batch multiple requests for optimal processing."""

    def __init__(self, batch_size: int = 10, batch_timeout_ms: int = 100):
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self._pending_requests: List[Dict[str, Any]] = []
        self._batch_processors: Dict[str, Callable] = {}
        self._batch_lock = asyncio.Lock()

    def register_batch_processor(self, endpoint_pattern: str, processor: Callable) -> None:
        """Register a batch processor for specific endpoints."""
        self._batch_processors[endpoint_pattern] = processor

    async def add_to_batch(
        self, request_id: str, endpoint: str, request_data: Any, response_future: asyncio.Future
    ) -> None:
        """Add request to batch for processing."""
        async with self._batch_lock:
            self._pending_requests.append(
                {
                    "id": request_id,
                    "endpoint": endpoint,
                    "data": request_data,
                    "future": response_future,
                    "added_at": time.time(),
                }
            )

            # Check if we should process the batch
            if len(self._pending_requests) >= self.batch_size:
                await self._process_batch()

    async def _process_batch(self) -> None:
        """Process current batch of requests."""
        if not self._pending_requests:
            return

        # Group requests by endpoint
        batches_by_endpoint = {}
        for request in self._pending_requests:
            endpoint = request["endpoint"]
            if endpoint not in batches_by_endpoint:
                batches_by_endpoint[endpoint] = []
            batches_by_endpoint[endpoint].append(request)

        # Process each endpoint batch
        for endpoint, requests in batches_by_endpoint.items():
            processor = self._find_batch_processor(endpoint)
            if processor:
                try:
                    # Extract request data
                    request_data = [req["data"] for req in requests]

                    # Process batch
                    results = await processor(request_data)

                    # Set results for individual futures
                    for i, request in enumerate(requests):
                        if i < len(results):
                            request["future"].set_result(results[i])
                        else:
                            request["future"].set_exception(
                                Exception("Batch processing failed for request")
                            )

                except Exception as e:
                    # Set exception for all requests in batch
                    for request in requests:
                        request["future"].set_exception(e)
            else:
                # No batch processor found, fail requests
                for request in requests:
                    request["future"].set_exception(
                        Exception(f"No batch processor for endpoint: {endpoint}")
                    )

        # Clear processed requests
        self._pending_requests.clear()

    def _find_batch_processor(self, endpoint: str) -> Optional[Callable]:
        """Find batch processor for endpoint."""
        for pattern, processor in self._batch_processors.items():
            if pattern in endpoint:
                return processor
        return None

    async def start_batch_timer(self) -> None:
        """Start batch processing timer."""
        asyncio.create_task(self._batch_timer())

    async def _batch_timer(self) -> None:
        """Timer to process batches periodically."""
        while True:
            await asyncio.sleep(self.batch_timeout_ms / 1000)

            async with self._batch_lock:
                if self._pending_requests:
                    # Check if oldest request has exceeded timeout
                    current_time = time.time()
                    oldest_request_time = min(req["added_at"] for req in self._pending_requests)

                    if (current_time - oldest_request_time) * 1000 >= self.batch_timeout_ms:
                        await self._process_batch()


class APIOptimizer:
    """Comprehensive API response optimization."""

    def __init__(
        self, cache_manager: CacheManager, metrics_collector: Optional[MetricsCollector] = None
    ):
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector

        # Components
        self.compressor = ResponseCompressor()
        self.response_cache = ResponseCache(cache_manager)
        self.request_batcher = RequestBatcher()

        # Metrics
        self.metrics = ResponseMetrics()

        # Configuration
        self._enable_compression = True
        self._enable_caching = True
        self._enable_batching = True

        # Response size tracking for metrics
        self._response_sizes = []
        self._compression_ratios = []

    async def initialize(self) -> None:
        """Initialize API optimizer."""
        # Start batch timer
        if self._enable_batching:
            await self.request_batcher.start_batch_timer()

        # Register default cache patterns
        self._register_default_cache_patterns()

        print("API optimizer initialized")

    def _register_default_cache_patterns(self) -> None:
        """Register default cache key patterns."""

        # Analytics cache pattern
        def analytics_cache_key(request: Request, context: Optional[Dict] = None) -> str:
            # Include test_id and date range in cache key
            test_id = request.path_params.get("test_id", "")
            query_params = dict(request.query_params)

            key_parts = ["analytics", test_id, json.dumps(query_params, sort_keys=True)]

            if context and "user_id" in context:
                key_parts.append(f"user_{context['user_id']}")

            return hashlib.sha256("|".join(key_parts).encode()).hexdigest()[:16]

        self.response_cache.register_cache_pattern("/analytics/", analytics_cache_key)

        # Provider data cache pattern
        def provider_cache_key(request: Request, context: Optional[Dict] = None) -> str:
            provider_id = request.path_params.get("provider_id", "")
            return f"provider_{provider_id}_{hashlib.sha256(str(request.url).encode()).hexdigest()[:8]}"

        self.response_cache.register_cache_pattern("/providers/", provider_cache_key)

    @asynccontextmanager
    async def optimize_response(
        self,
        request: Request,
        cache_ttl: Optional[int] = None,
        enable_compression: Optional[bool] = None,
        enable_caching: Optional[bool] = None,
        additional_context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Context manager for optimizing API responses."""
        start_time = time.time()

        # Configuration overrides
        use_compression = (
            enable_compression if enable_compression is not None else self._enable_compression
        )
        use_caching = enable_caching if enable_caching is not None else self._enable_caching

        optimization_info = {
            "cache_hit": False,
            "compression_used": False,
            "compression_ratio": 1.0,
            "original_size": 0,
            "final_size": 0,
            "processing_time_ms": 0.0,
        }

        # Try cache first
        if use_caching:
            cached_response = await self.response_cache.get_cached_response(
                request, additional_context
            )

            if cached_response:
                optimization_info["cache_hit"] = True
                self.metrics.cached_responses += 1

                yield {
                    "data": cached_response["data"],
                    "optimization_info": optimization_info,
                    "from_cache": True,
                }
                return

        # Cache miss - prepare for response generation
        response_data = {}

        try:
            yield {
                "set_response": lambda data: response_data.update({"data": data}),
                "optimization_info": optimization_info,
            }

            # Process response after generation
            if "data" in response_data:
                processing_time = (time.time() - start_time) * 1000
                optimization_info["processing_time_ms"] = processing_time

                # Cache the response
                if use_caching:
                    await self.response_cache.cache_response(
                        request, response_data["data"], cache_ttl, additional_context
                    )

                # Update metrics
                self.metrics.total_responses += 1
                self._update_response_metrics(optimization_info)

        except Exception as e:
            print(f"Error in response optimization: {e}")
            raise

    def create_optimized_response(
        self,
        data: Any,
        request: Request,
        status_code: int = 200,
        headers: Optional[Dict[str, str]] = None,
    ) -> Response:
        """Create optimized response with compression."""
        headers = headers or {}

        # Serialize data
        if isinstance(data, (dict, list)):
            content = json.dumps(data, separators=(",", ":")).encode("utf-8")
            content_type = "application/json"
        else:
            content = str(data).encode("utf-8")
            content_type = "text/plain"

        optimization_info = {
            "original_size": len(content),
            "compression_used": False,
            "compression_ratio": 1.0,
        }

        # Apply compression if enabled
        if self._enable_compression:
            accept_encoding = request.headers.get("accept-encoding", "")
            algorithm = self.compressor.select_compression_algorithm(
                accept_encoding, content_type, len(content)
            )

            if algorithm != CompressionType.NONE:
                compressed_content, compression_time, compression_ratio = (
                    self.compressor.compress_response(content, algorithm)
                )

                if compressed_content != content:  # Compression was applied
                    content = compressed_content
                    headers["Content-Encoding"] = self.compressor.get_content_encoding(algorithm)
                    optimization_info["compression_used"] = True
                    optimization_info["compression_ratio"] = compression_ratio

                    # Update metrics
                    self.metrics.compressed_responses += 1
                    self.metrics.compression_time_ms += compression_time

        # Set headers
        headers["Content-Type"] = content_type
        headers["Content-Length"] = str(len(content))
        headers["X-Compression-Ratio"] = str(optimization_info["compression_ratio"])

        # Update metrics
        optimization_info["final_size"] = len(content)
        self._update_response_metrics(optimization_info)

        return Response(content=content, status_code=status_code, headers=headers)

    def _update_response_metrics(self, optimization_info: Dict[str, Any]) -> None:
        """Update response metrics."""
        original_size = optimization_info.get("original_size", 0)
        compression_ratio = optimization_info.get("compression_ratio", 1.0)

        # Track response sizes
        self._response_sizes.append(original_size)
        if len(self._response_sizes) > 1000:
            self._response_sizes = self._response_sizes[-1000:]

        # Track compression ratios
        if compression_ratio > 1.0:
            self._compression_ratios.append(compression_ratio)
            if len(self._compression_ratios) > 1000:
                self._compression_ratios = self._compression_ratios[-1000:]

        # Update averages
        if self._response_sizes:
            self.metrics.average_response_size_bytes = sum(self._response_sizes) / len(
                self._response_sizes
            )

        if self._compression_ratios:
            self.metrics.average_compression_ratio = sum(self._compression_ratios) / len(
                self._compression_ratios
            )

        # Update cache hit rate
        if self.metrics.total_responses > 0:
            self.metrics.cache_hit_rate = (
                self.metrics.cached_responses / self.metrics.total_responses
            )

        self.metrics.last_updated = datetime.utcnow()

        # Record to external metrics collector
        if self.metrics_collector:
            self.metrics_collector.set_custom_gauge("api_response_size_bytes", original_size)
            if compression_ratio > 1.0:
                self.metrics_collector.set_custom_gauge("api_compression_ratio", compression_ratio)

    async def batch_process_requests(
        self, endpoint_pattern: str, processor: Callable[[List[Any]], List[Any]]
    ) -> None:
        """Register a batch processor for specific endpoint."""
        self.request_batcher.register_batch_processor(endpoint_pattern, processor)

    async def submit_to_batch(self, request_id: str, endpoint: str, request_data: Any) -> Any:
        """Submit request to batch processor."""
        if not self._enable_batching:
            raise ValueError("Batching is not enabled")

        future = asyncio.Future()
        await self.request_batcher.add_to_batch(request_id, endpoint, request_data, future)

        # Increment batch counter
        self.metrics.batched_requests += 1

        return await future

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get API optimization statistics."""
        return {
            "total_responses": self.metrics.total_responses,
            "cached_responses": self.metrics.cached_responses,
            "compressed_responses": self.metrics.compressed_responses,
            "cache_hit_rate": self.metrics.cache_hit_rate,
            "compression_stats": {
                "compression_enabled": self._enable_compression,
                "average_compression_ratio": self.metrics.average_compression_ratio,
                "total_compression_time_ms": self.metrics.compression_time_ms,
                "compression_rate": (
                    self.metrics.compressed_responses / max(self.metrics.total_responses, 1)
                ),
            },
            "response_stats": {
                "average_response_size_bytes": self.metrics.average_response_size_bytes,
                "total_responses": self.metrics.total_responses,
            },
            "batching_stats": {
                "batching_enabled": self._enable_batching,
                "batched_requests": self.metrics.batched_requests,
            },
            "performance": {
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "average_compression_ratio": self.metrics.average_compression_ratio,
            },
            "last_updated": self.metrics.last_updated.isoformat(),
        }

    async def optimize_api_configuration(self) -> Dict[str, Any]:
        """Analyze and suggest API optimization configurations."""
        stats = self.get_optimization_stats()
        suggestions = []

        # Cache optimization suggestions
        if stats["cache_hit_rate"] < 0.3:
            suggestions.append(
                "Low cache hit rate - consider increasing cache TTL or improving cache key strategy"
            )

        # Compression optimization suggestions
        compression_rate = stats["compression_stats"]["compression_rate"]
        if compression_rate < 0.5 and self._enable_compression:
            suggestions.append(
                "Low compression rate - check content types and compression thresholds"
            )

        avg_compression_ratio = stats["compression_stats"]["average_compression_ratio"]
        if avg_compression_ratio < 2.0 and compression_rate > 0:
            suggestions.append("Low compression efficiency - consider adjusting compression level")

        # Response size suggestions
        avg_response_size = stats["response_stats"]["average_response_size_bytes"]
        if avg_response_size > 100000:  # 100KB
            suggestions.append(
                "Large average response size - consider pagination or response filtering"
            )

        # Batching suggestions
        if self.metrics.batched_requests == 0 and self._enable_batching:
            suggestions.append(
                "Batching enabled but not used - consider implementing batch endpoints"
            )

        return {
            "current_stats": stats,
            "suggestions": suggestions,
            "recommended_actions": [
                "Monitor cache hit rates and adjust TTL as needed",
                "Review compression algorithms and thresholds",
                "Implement response pagination for large datasets",
                "Consider implementing GraphQL for flexible queries",
                "Monitor and optimize database query patterns",
            ],
        }

"""Performance optimization middleware for FastAPI."""

import asyncio
import gzip
import time
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlencode

from fastapi import Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from ....infrastructure.performance.cache_manager import CacheLayer
from ....infrastructure.performance.performance_manager import get_performance_manager


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Middleware for performance optimization including caching, compression, and monitoring."""

    def __init__(
        self,
        app: ASGIApp,
        enable_caching: bool = True,
        enable_compression: bool = True,
        enable_metrics: bool = True,
        compression_threshold: int = 1024,
        cache_ttl_default: int = 300,
        cacheable_methods: set[str] = None,
        cacheable_status_codes: set[int] = None,
    ):
        super().__init__(app)
        self.enable_caching = enable_caching
        self.enable_compression = enable_compression
        self.enable_metrics = enable_metrics
        self.compression_threshold = compression_threshold
        self.cache_ttl_default = cache_ttl_default
        self.cacheable_methods = cacheable_methods or {"GET", "HEAD"}
        self.cacheable_status_codes = cacheable_status_codes or {200, 201, 202, 203, 204}

        # Cache configuration for different endpoints
        self.endpoint_cache_config = {
            "/api/v1/tests": {"ttl": 120, "vary": ["Authorization"]},
            "/api/v1/providers": {"ttl": 600, "vary": ["Authorization"]},
            "/api/v1/analytics": {"ttl": 60, "vary": ["Authorization"]},
            "/health": {"ttl": 30, "vary": []},
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with performance optimizations."""
        start_time = time.time()

        # Get performance manager
        perf_manager = get_performance_manager()
        if not perf_manager:
            # No performance manager available, proceed normally
            response = await call_next(request)
            return response

        # Generate cache key if caching is enabled
        cache_key = None
        if self.enable_caching and self._is_cacheable_request(request):
            cache_key = await self._generate_cache_key(request)

            # Try to get cached response
            cached_response = await self._get_cached_response(cache_key, request.url.path)
            if cached_response:
                # Add cache hit headers
                cached_response.headers["X-Cache"] = "HIT"
                cached_response.headers["X-Cache-Key"] = cache_key[:16] + "..."

                # Record metrics
                if self.enable_metrics and perf_manager.metrics_collector:
                    perf_manager.metrics_collector.increment_custom_counter(
                        "http_cache_hits_total", labels={"endpoint": request.url.path}
                    )

                return cached_response

        # Process request
        try:
            response = await call_next(request)

            # Measure response time
            response_time = time.time() - start_time

            # Record metrics
            if self.enable_metrics and perf_manager.metrics_collector:
                perf_manager.metrics_collector.record_custom_timer(
                    "http_request_duration_seconds",
                    response_time,
                    labels={
                        "method": request.method,
                        "endpoint": request.url.path,
                        "status_code": str(response.status_code),
                    },
                )

                # Cache miss counter
                if cache_key:
                    perf_manager.metrics_collector.increment_custom_counter(
                        "http_cache_misses_total", labels={"endpoint": request.url.path}
                    )

            # Add performance headers
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            response.headers["X-Cache"] = "MISS" if cache_key else "DISABLED"

            # Apply compression if enabled
            if self.enable_compression:
                response = await self._apply_compression(request, response)

            # Cache response if appropriate
            if cache_key and self._is_cacheable_response(response):
                await self._cache_response(cache_key, response, request.url.path)

            return response

        except Exception as e:
            # Record error metrics
            if self.enable_metrics and perf_manager.metrics_collector:
                perf_manager.metrics_collector.increment_custom_counter(
                    "http_requests_errors_total",
                    labels={
                        "method": request.method,
                        "endpoint": request.url.path,
                        "error_type": type(e).__name__,
                    },
                )
            raise

    def _is_cacheable_request(self, request: Request) -> bool:
        """Check if request is cacheable."""
        # Only cache safe methods
        if request.method not in self.cacheable_methods:
            return False

        # Don't cache requests with authentication if not configured
        auth_header = request.headers.get("Authorization")
        if auth_header and not any(
            request.url.path.startswith(endpoint) for endpoint in self.endpoint_cache_config.keys()
        ):
            return False

        # Don't cache requests with cache-control: no-cache
        cache_control = request.headers.get("Cache-Control", "")
        if "no-cache" in cache_control.lower():
            return False

        return True

    def _is_cacheable_response(self, response: Response) -> bool:
        """Check if response is cacheable."""
        # Only cache successful responses
        if response.status_code not in self.cacheable_status_codes:
            return False

        # Check cache-control headers
        cache_control = response.headers.get("Cache-Control", "")
        if "no-cache" in cache_control.lower() or "no-store" in cache_control.lower():
            return False

        return True

    async def _generate_cache_key(self, request: Request) -> str:
        """Generate cache key for request."""
        # Base components
        method = request.method
        path = str(request.url.path)
        query = str(request.url.query) if request.url.query else ""

        # Get endpoint-specific vary headers
        endpoint_config = self._get_endpoint_cache_config(path)
        vary_headers = endpoint_config.get("vary", [])

        # Include varying headers in cache key
        header_parts = []
        for header_name in vary_headers:
            header_value = request.headers.get(header_name, "")
            if header_name == "Authorization" and header_value:
                # Hash authorization header for privacy
                import hashlib

                header_value = hashlib.md5(header_value.encode()).hexdigest()[:8]
            header_parts.append(f"{header_name}:{header_value}")

        # Combine all parts
        key_parts = [method, path, query] + header_parts
        cache_key = "|".join(key_parts)

        # Hash for consistent length
        import hashlib

        return hashlib.sha256(cache_key.encode()).hexdigest()

    async def _get_cached_response(self, cache_key: str, endpoint: str) -> Optional[Response]:
        """Get cached response if available."""
        perf_manager = get_performance_manager()
        if not perf_manager or not perf_manager.cache_manager:
            return None

        try:
            cached_data = await perf_manager.cache_manager.get(
                cache_key, namespace="http_responses", layer=CacheLayer.HYBRID
            )

            if cached_data:
                # Reconstruct response
                return JSONResponse(
                    content=cached_data["content"],
                    status_code=cached_data["status_code"],
                    headers=cached_data.get("headers", {}),
                )

        except Exception as e:
            print(f"Error getting cached response: {e}")

        return None

    async def _cache_response(self, cache_key: str, response: Response, endpoint: str) -> None:
        """Cache response data."""
        perf_manager = get_performance_manager()
        if not perf_manager or not perf_manager.cache_manager:
            return

        try:
            # Only cache JSON responses for now
            if not isinstance(response, JSONResponse):
                return

            # Get endpoint-specific configuration
            endpoint_config = self._get_endpoint_cache_config(endpoint)
            ttl = endpoint_config.get("ttl", self.cache_ttl_default)

            # Prepare cache data
            cache_data = {
                "content": response.body.decode() if hasattr(response, "body") else None,
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "cached_at": time.time(),
            }

            # Cache the response
            await perf_manager.cache_manager.set(
                cache_key, cache_data, ttl=ttl, namespace="http_responses", layer=CacheLayer.HYBRID
            )

        except Exception as e:
            print(f"Error caching response: {e}")

    def _get_endpoint_cache_config(self, path: str) -> Dict[str, Any]:
        """Get cache configuration for endpoint."""
        for endpoint, config in self.endpoint_cache_config.items():
            if path.startswith(endpoint):
                return config

        return {"ttl": self.cache_ttl_default, "vary": []}

    async def _apply_compression(self, request: Request, response: Response) -> Response:
        """Apply gzip compression to response if appropriate."""
        # Check if client accepts gzip
        accept_encoding = request.headers.get("Accept-Encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response

        # Don't compress if already compressed
        if response.headers.get("Content-Encoding"):
            return response

        # Get response content
        if isinstance(response, StreamingResponse):
            # Don't compress streaming responses
            return response

        # Get response body
        if hasattr(response, "body"):
            content = response.body
        elif isinstance(response, JSONResponse):
            content = response.render(response.content).encode("utf-8")
        else:
            return response

        # Only compress if content is large enough
        if len(content) < self.compression_threshold:
            return response

        # Compress content
        compressed_content = gzip.compress(content)

        # Create new response with compressed content
        new_response = Response(
            content=compressed_content,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

        # Add compression headers
        new_response.headers["Content-Encoding"] = "gzip"
        new_response.headers["Content-Length"] = str(len(compressed_content))
        new_response.headers["Vary"] = "Accept-Encoding"

        return new_response


class RequestBatchingMiddleware(BaseHTTPMiddleware):
    """Middleware for batching similar requests to optimize performance."""

    def __init__(
        self,
        app: ASGIApp,
        batch_timeout_ms: int = 50,
        max_batch_size: int = 100,
        enable_batching: bool = True,
    ):
        super().__init__(app)
        self.batch_timeout_ms = batch_timeout_ms
        self.max_batch_size = max_batch_size
        self.enable_batching = enable_batching

        # Request batching state
        self._batch_queues: Dict[str, list] = {}
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._batch_results: Dict[str, Any] = {}

        # Batchable endpoints configuration
        self.batchable_endpoints = {
            "/api/v1/providers/models": {"timeout_ms": 100, "max_size": 50},
            "/api/v1/analytics/metrics": {"timeout_ms": 200, "max_size": 20},
        }

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with batching optimization."""
        if not self.enable_batching or not self._is_batchable_request(request):
            # Process normally
            return await call_next(request)

        # Generate batch key
        batch_key = self._generate_batch_key(request)

        # Add to batch queue
        request_id = id(request)
        batch_item = {
            "id": request_id,
            "request": request,
            "call_next": call_next,
            "future": asyncio.Future(),
        }

        if batch_key not in self._batch_queues:
            self._batch_queues[batch_key] = []

        self._batch_queues[batch_key].append(batch_item)

        # Set up batch timer if needed
        if batch_key not in self._batch_timers:
            endpoint_config = self._get_batching_config(request.url.path)
            timeout_ms = endpoint_config.get("timeout_ms", self.batch_timeout_ms)

            self._batch_timers[batch_key] = asyncio.create_task(
                self._batch_timer(batch_key, timeout_ms / 1000.0)
            )

        # Check if batch is full
        endpoint_config = self._get_batching_config(request.url.path)
        max_size = endpoint_config.get("max_size", self.max_batch_size)

        if len(self._batch_queues[batch_key]) >= max_size:
            # Process batch immediately
            await self._process_batch(batch_key)

        # Wait for result
        return await batch_item["future"]

    def _is_batchable_request(self, request: Request) -> bool:
        """Check if request can be batched."""
        # Only batch GET requests for now
        if request.method != "GET":
            return False

        # Check if endpoint supports batching
        return any(
            request.url.path.startswith(endpoint) for endpoint in self.batchable_endpoints.keys()
        )

    def _generate_batch_key(self, request: Request) -> str:
        """Generate key for batching similar requests."""
        # Group by endpoint and common query parameters
        path = request.url.path

        # For analytics endpoints, group by time range
        if path.startswith("/api/v1/analytics"):
            query_params = dict(request.query_params)
            time_range = query_params.get("time_range", "")
            return f"{path}|time_range:{time_range}"

        # For model endpoints, group by provider
        if path.startswith("/api/v1/providers"):
            query_params = dict(request.query_params)
            provider = query_params.get("provider", "")
            return f"{path}|provider:{provider}"

        return path

    def _get_batching_config(self, path: str) -> Dict[str, Any]:
        """Get batching configuration for endpoint."""
        for endpoint, config in self.batchable_endpoints.items():
            if path.startswith(endpoint):
                return config

        return {"timeout_ms": self.batch_timeout_ms, "max_size": self.max_batch_size}

    async def _batch_timer(self, batch_key: str, timeout_seconds: float) -> None:
        """Timer for batch processing."""
        try:
            await asyncio.sleep(timeout_seconds)
            await self._process_batch(batch_key)
        except asyncio.CancelledError:
            pass

    async def _process_batch(self, batch_key: str) -> None:
        """Process a batch of requests."""
        if batch_key not in self._batch_queues:
            return

        batch_items = self._batch_queues.pop(batch_key, [])
        if not batch_items:
            return

        # Cancel timer
        if batch_key in self._batch_timers:
            self._batch_timers[batch_key].cancel()
            del self._batch_timers[batch_key]

        # Process all requests in batch
        try:
            # For now, process sequentially (could be optimized to true batching)
            for item in batch_items:
                try:
                    response = await item["call_next"](item["request"])
                    item["future"].set_result(response)
                except Exception as e:
                    item["future"].set_exception(e)

        except Exception as e:
            # Set exception for all items
            for item in batch_items:
                if not item["future"].done():
                    item["future"].set_exception(e)

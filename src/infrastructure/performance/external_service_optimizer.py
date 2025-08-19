"""Enhanced external service optimization with intelligent batching and caching."""

import asyncio
import json
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import httpx

from .cache_manager import CacheLayer, CacheManager
from .circuit_breaker_manager import CircuitBreakerManager
from .connection_optimizer import ConnectionOptimizer
from .metrics_collector import MetricsCollector


class RequestPriority(Enum):
    """Request priority levels for intelligent queuing."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BatchConfig:
    """Configuration for request batching."""

    max_batch_size: int = 10
    batch_timeout_ms: int = 100
    enable_adaptive_batching: bool = True
    priority_based_batching: bool = True


@dataclass
class ExternalServiceConfig:
    """Configuration for external service optimization."""

    service_name: str
    base_url: str
    timeout_seconds: float = 30.0
    max_retries: int = 3
    retry_backoff_factor: float = 1.5
    circuit_breaker_enabled: bool = True
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    batch_config: BatchConfig = field(default_factory=BatchConfig)
    enable_compression: bool = True
    enable_http2: bool = True


@dataclass
class RequestMetrics:
    """Metrics for external service requests."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    cached_requests: int = 0
    batched_requests: int = 0
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)


class ExternalServiceOptimizer:
    """Advanced optimizer for external service calls with intelligent batching and caching."""

    def __init__(
        self,
        connection_optimizer: ConnectionOptimizer,
        cache_manager: Optional[CacheManager] = None,
        circuit_breaker_manager: Optional[CircuitBreakerManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.connection_optimizer = connection_optimizer
        self.cache_manager = cache_manager
        self.circuit_breaker_manager = circuit_breaker_manager
        self.metrics_collector = metrics_collector

        # Service configurations
        self._service_configs: Dict[str, ExternalServiceConfig] = {}

        # Request batching
        self._batch_queues: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._batch_timers: Dict[str, asyncio.Task] = {}
        self._batch_locks: Dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

        # Request metrics
        self._service_metrics: Dict[str, RequestMetrics] = defaultdict(RequestMetrics)
        self._latency_windows: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Cache keys for different request types
        self._cache_strategies: Dict[str, Callable] = {
            "model_request": self._generate_model_request_cache_key,
            "analytics_request": self._generate_analytics_cache_key,
            "default": self._generate_default_cache_key,
        }

    def register_service(self, config: ExternalServiceConfig) -> None:
        """Register a new external service configuration."""
        self._service_configs[config.service_name] = config
        self._service_metrics[config.service_name] = RequestMetrics()

    @asynccontextmanager
    async def optimized_request(
        self,
        service_name: str,
        method: str = "POST",
        endpoint: str = "/",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        cache_key: Optional[str] = None,
        cache_strategy: str = "default",
        enable_batching: bool = True,
    ):
        """Context manager for optimized external service requests."""
        config = self._service_configs.get(service_name)
        if not config:
            raise ValueError(f"Service {service_name} not registered")

        start_time = time.time()
        request_id = f"{service_name}_{int(time.time() * 1000000)}"

        try:
            # Try cache first if enabled
            if config.cache_enabled and cache_key:
                cached_response = await self._get_cached_response(service_name, cache_key)
                if cached_response:
                    self._record_cache_hit(service_name, time.time() - start_time)
                    yield cached_response
                    return

            # Prepare request context
            request_context = {
                "service_name": service_name,
                "method": method,
                "endpoint": endpoint,
                "data": data,
                "headers": headers or {},
                "priority": priority,
                "cache_key": cache_key,
                "cache_strategy": cache_strategy,
                "request_id": request_id,
                "start_time": start_time,
                "future": asyncio.Future(),
            }

            # Handle batching if enabled and applicable
            if enable_batching and self._is_batchable_request(service_name, method, endpoint):
                await self._handle_batched_request(request_context)
                response = await request_context["future"]
            else:
                # Execute single request
                response = await self._execute_single_request(config, request_context)

            # Cache response if applicable
            if config.cache_enabled and cache_key and response.get("success"):
                await self._cache_response(
                    service_name, cache_key, response, config.cache_ttl_seconds
                )

            # Record metrics
            self._record_request_metrics(service_name, start_time, True, response)

            yield response

        except Exception as e:
            self._record_request_metrics(service_name, start_time, False, {"error": str(e)})
            raise

    async def _execute_single_request(
        self, config: ExternalServiceConfig, request_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single HTTP request with optimizations."""

        # Get optimized HTTP client
        client_type = "model_provider" if "model" in config.service_name.lower() else "default"

        async with self.connection_optimizer.monitored_http_request(client_type) as client:
            # Prepare request
            url = f"{config.base_url.rstrip('/')}{request_context['endpoint']}"
            headers = request_context.get("headers", {})

            # Add compression if enabled
            if config.enable_compression:
                headers["Accept-Encoding"] = "gzip, deflate"

            # Prepare request data
            request_data = request_context.get("data")
            if request_data and isinstance(request_data, dict):
                headers["Content-Type"] = "application/json"
                request_data = json.dumps(request_data)

            # Execute with circuit breaker if enabled
            if config.circuit_breaker_enabled and self.circuit_breaker_manager:
                circuit_breaker = await self.circuit_breaker_manager.get_circuit_breaker(
                    f"{config.service_name}_http"
                )
                response = await circuit_breaker.execute(
                    lambda: self._make_http_request(
                        client, request_context["method"], url, request_data, headers, config
                    )
                )
            else:
                response = await self._make_http_request(
                    client, request_context["method"], url, request_data, headers, config
                )

            return response

    async def _make_http_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        data: Optional[str],
        headers: Dict[str, str],
        config: ExternalServiceConfig,
    ) -> Dict[str, Any]:
        """Make the actual HTTP request with retry logic."""

        for attempt in range(config.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = await client.get(url, headers=headers)
                elif method.upper() == "POST":
                    response = await client.post(url, content=data, headers=headers)
                elif method.upper() == "PUT":
                    response = await client.put(url, content=data, headers=headers)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()

                # Parse response
                if response.headers.get("content-type", "").startswith("application/json"):
                    response_data = response.json()
                else:
                    response_data = {"content": response.text}

                return {
                    "success": True,
                    "status_code": response.status_code,
                    "data": response_data,
                    "headers": dict(response.headers),
                    "attempt": attempt + 1,
                }

            except httpx.HTTPStatusError as e:
                if attempt == config.max_retries:
                    return {
                        "success": False,
                        "error": f"HTTP {e.response.status_code}: {e.response.text}",
                        "status_code": e.response.status_code,
                        "attempt": attempt + 1,
                    }

                # Don't retry client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    return {
                        "success": False,
                        "error": f"HTTP {e.response.status_code}: {e.response.text}",
                        "status_code": e.response.status_code,
                        "attempt": attempt + 1,
                    }

            except Exception as e:
                if attempt == config.max_retries:
                    return {"success": False, "error": str(e), "attempt": attempt + 1}

            # Wait before retry with exponential backoff
            if attempt < config.max_retries:
                wait_time = config.retry_backoff_factor**attempt
                await asyncio.sleep(wait_time)

        return {"success": False, "error": "Max retries exceeded"}

    def _is_batchable_request(self, service_name: str, method: str, endpoint: str) -> bool:
        """Check if request can be batched."""
        config = self._service_configs.get(service_name)
        if not config or not config.batch_config:
            return False

        # Only batch POST requests to specific endpoints
        if method.upper() != "POST":
            return False

        # Batchable endpoints (can be configured per service)
        batchable_patterns = [
            "/v1/chat/completions",  # OpenAI-style endpoints
            "/v1/completions",
            "/api/completion",
            "/generate",
            "/inference",
        ]

        return any(pattern in endpoint for pattern in batchable_patterns)

    async def _handle_batched_request(self, request_context: Dict[str, Any]) -> None:
        """Handle request batching logic."""
        service_name = request_context["service_name"]
        config = self._service_configs[service_name]

        # Generate batch key based on endpoint and priority
        batch_key = (
            f"{service_name}_{request_context['endpoint']}_{request_context['priority'].value}"
        )

        async with self._batch_locks[batch_key]:
            # Add to batch queue
            self._batch_queues[batch_key].append(request_context)

            # Check if batch is full
            if len(self._batch_queues[batch_key]) >= config.batch_config.max_batch_size:
                await self._process_batch(batch_key)
            elif batch_key not in self._batch_timers:
                # Set batch timer
                self._batch_timers[batch_key] = asyncio.create_task(
                    self._batch_timer(batch_key, config.batch_config.batch_timeout_ms / 1000.0)
                )

    async def _batch_timer(self, batch_key: str, timeout_seconds: float) -> None:
        """Timer for batch processing."""
        try:
            await asyncio.sleep(timeout_seconds)
            await self._process_batch(batch_key)
        except asyncio.CancelledError:
            pass

    async def _process_batch(self, batch_key: str) -> None:
        """Process a batch of requests."""
        async with self._batch_locks[batch_key]:
            if batch_key not in self._batch_queues or not self._batch_queues[batch_key]:
                return

            batch_requests = self._batch_queues[batch_key].copy()
            self._batch_queues[batch_key].clear()

            # Cancel timer
            if batch_key in self._batch_timers:
                self._batch_timers[batch_key].cancel()
                del self._batch_timers[batch_key]

        if not batch_requests:
            return

        # Process batch
        service_name = batch_requests[0]["service_name"]
        config = self._service_configs[service_name]

        try:
            # Check if service supports true batching
            if await self._service_supports_batching(service_name):
                # Create batch request
                batch_response = await self._execute_batch_request(config, batch_requests)

                # Distribute responses
                for i, request_context in enumerate(batch_requests):
                    if i < len(batch_response.get("responses", [])):
                        request_context["future"].set_result(batch_response["responses"][i])
                    else:
                        request_context["future"].set_result(
                            {"success": False, "error": "Batch response missing"}
                        )
            else:
                # Execute requests in parallel (pseudo-batching)
                tasks = [
                    self._execute_single_request(config, req_ctx) for req_ctx in batch_requests
                ]

                responses = await asyncio.gather(*tasks, return_exceptions=True)

                for i, (request_context, response) in enumerate(zip(batch_requests, responses)):
                    if isinstance(response, Exception):
                        request_context["future"].set_exception(response)
                    else:
                        request_context["future"].set_result(response)

        except Exception as e:
            # Set exception for all requests
            for request_context in batch_requests:
                if not request_context["future"].done():
                    request_context["future"].set_exception(e)

    async def _service_supports_batching(self, service_name: str) -> bool:
        """Check if service supports true batching."""
        # This would be configurable per service
        # For now, assume OpenAI-style services support batching
        return "openai" in service_name.lower() or "batch" in service_name.lower()

    async def _execute_batch_request(
        self, config: ExternalServiceConfig, batch_requests: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Execute a true batch request to the service."""
        # Combine individual requests into batch format
        batch_data = {
            "requests": [{"id": req["request_id"], "data": req["data"]} for req in batch_requests]
        }

        # Execute batch request
        batch_context = {
            "service_name": config.service_name,
            "method": "POST",
            "endpoint": "/batch",  # Assume batch endpoint
            "data": batch_data,
            "headers": {"Content-Type": "application/json"},
            "request_id": f"batch_{int(time.time())}",
            "start_time": time.time(),
        }

        return await self._execute_single_request(config, batch_context)

    async def _get_cached_response(
        self, service_name: str, cache_key: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached response if available."""
        if not self.cache_manager:
            return None

        try:
            return await self.cache_manager.get(
                cache_key, namespace=f"external_service_{service_name}", layer=CacheLayer.HYBRID
            )
        except Exception:
            return None

    async def _cache_response(
        self, service_name: str, cache_key: str, response: Dict[str, Any], ttl_seconds: int
    ) -> None:
        """Cache successful response."""
        if not self.cache_manager or not response.get("success"):
            return

        try:
            await self.cache_manager.set(
                cache_key,
                response,
                ttl=ttl_seconds,
                namespace=f"external_service_{service_name}",
                layer=CacheLayer.HYBRID,
            )
        except Exception:
            pass  # Don't fail request due to cache errors

    def _generate_model_request_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for model requests."""
        import hashlib

        # Include model, prompt, and key parameters
        key_data = {
            "model": data.get("model", ""),
            "prompt": data.get("prompt", ""),
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_tokens", 1000),
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _generate_analytics_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate cache key for analytics requests."""
        import hashlib

        key_data = {
            "query": data.get("query", ""),
            "filters": data.get("filters", {}),
            "time_range": data.get("time_range", ""),
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _generate_default_cache_key(self, data: Dict[str, Any]) -> str:
        """Generate default cache key."""
        import hashlib

        key_string = json.dumps(data, sort_keys=True) if data else "empty"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _record_cache_hit(self, service_name: str, duration_seconds: float) -> None:
        """Record cache hit metrics."""
        metrics = self._service_metrics[service_name]
        metrics.cached_requests += 1

        if self.metrics_collector:
            self.metrics_collector.increment_custom_counter(
                "external_service_cache_hits_total", labels={"service": service_name}
            )
            self.metrics_collector.record_custom_timer(
                "external_service_cache_response_time",
                duration_seconds,
                labels={"service": service_name},
            )

    def _record_request_metrics(
        self, service_name: str, start_time: float, success: bool, response: Dict[str, Any]
    ) -> None:
        """Record request metrics."""
        duration_ms = (time.time() - start_time) * 1000
        metrics = self._service_metrics[service_name]

        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1

        # Update latency metrics
        latency_window = self._latency_windows[service_name]
        latency_window.append(duration_ms)

        if latency_window:
            metrics.average_latency_ms = sum(latency_window) / len(latency_window)
            sorted_latencies = sorted(latency_window)
            p95_index = int(len(sorted_latencies) * 0.95)
            metrics.p95_latency_ms = (
                sorted_latencies[p95_index]
                if p95_index < len(sorted_latencies)
                else sorted_latencies[-1]
            )

        metrics.last_updated = datetime.utcnow()

        # Record in metrics collector
        if self.metrics_collector:
            self.metrics_collector.record_custom_timer(
                "external_service_request_duration_seconds",
                duration_ms / 1000,
                labels={"service": service_name, "success": str(success)},
            )

            if success:
                self.metrics_collector.increment_custom_counter(
                    "external_service_requests_successful_total", labels={"service": service_name}
                )
            else:
                self.metrics_collector.increment_custom_counter(
                    "external_service_requests_failed_total", labels={"service": service_name}
                )

    def get_service_metrics(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for a specific service or all services."""
        if service_name:
            if service_name not in self._service_metrics:
                return {}

            metrics = self._service_metrics[service_name]
            return {
                "service_name": service_name,
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "cached_requests": metrics.cached_requests,
                "batched_requests": metrics.batched_requests,
                "success_rate": metrics.successful_requests / max(metrics.total_requests, 1),
                "cache_hit_rate": metrics.cached_requests / max(metrics.total_requests, 1),
                "average_latency_ms": metrics.average_latency_ms,
                "p95_latency_ms": metrics.p95_latency_ms,
                "last_updated": metrics.last_updated.isoformat(),
            }
        else:
            return {
                service: self.get_service_metrics(service)
                for service in self._service_metrics.keys()
            }

    async def optimize_services(self) -> Dict[str, Any]:
        """Run optimization for all registered services."""
        optimization_results = {}

        for service_name, config in self._service_configs.items():
            service_results = {"optimizations_applied": []}

            metrics = self._service_metrics[service_name]

            # Analyze and optimize batch configuration
            if config.batch_config and metrics.total_requests > 100:
                success_rate = metrics.successful_requests / metrics.total_requests
                avg_latency = metrics.average_latency_ms

                # Adjust batch size based on performance
                if success_rate > 0.95 and avg_latency < 1000:  # High success, low latency
                    if config.batch_config.max_batch_size < 20:
                        config.batch_config.max_batch_size = min(
                            20, config.batch_config.max_batch_size + 5
                        )
                        service_results["optimizations_applied"].append("increased_batch_size")

                elif success_rate < 0.90 or avg_latency > 5000:  # Low success or high latency
                    if config.batch_config.max_batch_size > 5:
                        config.batch_config.max_batch_size = max(
                            5, config.batch_config.max_batch_size - 3
                        )
                        service_results["optimizations_applied"].append("decreased_batch_size")

                # Adjust batch timeout
                if avg_latency < 500:  # Fast responses
                    config.batch_config.batch_timeout_ms = max(
                        50, config.batch_config.batch_timeout_ms - 10
                    )
                    service_results["optimizations_applied"].append("reduced_batch_timeout")
                elif avg_latency > 2000:  # Slow responses
                    config.batch_config.batch_timeout_ms = min(
                        500, config.batch_config.batch_timeout_ms + 20
                    )
                    service_results["optimizations_applied"].append("increased_batch_timeout")

            optimization_results[service_name] = service_results

        return optimization_results

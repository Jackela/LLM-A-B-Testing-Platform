"""Performance optimization decorators for API routes."""

import asyncio
import hashlib
import json
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response

# Simple in-memory cache for demonstration
_ROUTE_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_STATS = {"hits": 0, "misses": 0, "sets": 0}


def generate_cache_key(request: Request, *args, **kwargs) -> str:
    """Generate cache key from request parameters."""
    key_parts = [
        request.method,
        str(request.url.path),
        str(request.query_params),
        str(sorted(kwargs.items())),
    ]
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached_route(ttl_seconds: int = 300, cache_key_func: Optional[Callable] = None):
    """Decorator for caching API route responses."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request if available
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                # No request found, skip caching
                return await func(*args, **kwargs)

            # Generate cache key
            if cache_key_func:
                cache_key = cache_key_func(request, *args, **kwargs)
            else:
                cache_key = generate_cache_key(request, *args, **kwargs)

            # Check cache
            if cache_key in _ROUTE_CACHE:
                cached_item = _ROUTE_CACHE[cache_key]

                # Check if not expired
                if time.time() < cached_item["expires_at"]:
                    _CACHE_STATS["hits"] += 1

                    # Add cache headers to response if it's a Response object
                    result = cached_item["data"]
                    if hasattr(result, "headers"):
                        result.headers["X-Cache-Status"] = "HIT"
                        result.headers["X-Cache-Key"] = cache_key[:16]

                    return result
                else:
                    # Expired, remove from cache
                    del _ROUTE_CACHE[cache_key]

            # Cache miss - execute function
            _CACHE_STATS["misses"] += 1
            start_time = time.time()

            result = await func(*args, **kwargs)

            execution_time = time.time() - start_time

            # Cache the result
            _ROUTE_CACHE[cache_key] = {
                "data": result,
                "expires_at": time.time() + ttl_seconds,
                "created_at": time.time(),
                "execution_time": execution_time,
            }
            _CACHE_STATS["sets"] += 1

            # Add cache headers if it's a Response object
            if hasattr(result, "headers"):
                result.headers["X-Cache-Status"] = "MISS"
                result.headers["X-Cache-Key"] = cache_key[:16]
                result.headers["X-Execution-Time"] = f"{execution_time * 1000:.1f}ms"

            # Cleanup old cache entries periodically
            if len(_ROUTE_CACHE) > 1000:  # Max 1000 cached items
                await _cleanup_cache()

            return result

        return wrapper

    return decorator


async def _cleanup_cache():
    """Clean up expired cache entries."""
    current_time = time.time()
    expired_keys = [key for key, item in _ROUTE_CACHE.items() if current_time >= item["expires_at"]]

    for key in expired_keys:
        del _ROUTE_CACHE[key]

    # If still too many items, remove oldest ones
    if len(_ROUTE_CACHE) > 500:
        sorted_items = sorted(_ROUTE_CACHE.items(), key=lambda x: x[1]["created_at"])
        items_to_remove = len(_ROUTE_CACHE) - 500

        for i in range(items_to_remove):
            del _ROUTE_CACHE[sorted_items[i][0]]


def performance_monitor(operation_name: str = ""):
    """Decorator for monitoring API route performance."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or func.__name__

            try:
                result = await func(*args, **kwargs)

                execution_time = time.time() - start_time

                # Log performance metrics
                print(f"⚡ {operation}: {execution_time * 1000:.1f}ms")

                # Add performance headers if it's a Response object
                if hasattr(result, "headers"):
                    result.headers["X-Performance-Monitored"] = "true"
                    result.headers["X-Operation-Time"] = f"{execution_time * 1000:.1f}ms"
                    result.headers["X-Operation-Name"] = operation

                return result

            except Exception as e:
                execution_time = time.time() - start_time
                print(f"❌ {operation}: {execution_time * 1000:.1f}ms (ERROR: {str(e)})")
                raise

        return wrapper

    return decorator


def rate_limit_protection(max_requests: int = 100, window_seconds: int = 60):
    """Simple rate limiting decorator."""
    request_counts: Dict[str, Dict[str, Any]] = {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request to get client IP
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                return await func(*args, **kwargs)

            client_ip = request.client.host if request.client else "unknown"
            current_time = time.time()

            # Clean up old entries
            if client_ip in request_counts:
                request_counts[client_ip]["requests"] = [
                    req_time
                    for req_time in request_counts[client_ip]["requests"]
                    if current_time - req_time < window_seconds
                ]
            else:
                request_counts[client_ip] = {"requests": []}

            # Check rate limit
            if len(request_counts[client_ip]["requests"]) >= max_requests:
                from fastapi import HTTPException

                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds",
                )

            # Add current request
            request_counts[client_ip]["requests"].append(current_time)

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def get_cache_stats() -> Dict[str, Any]:
    """Get cache performance statistics."""
    total_requests = _CACHE_STATS["hits"] + _CACHE_STATS["misses"]
    hit_rate = _CACHE_STATS["hits"] / total_requests if total_requests > 0 else 0

    return {
        "cache_size": len(_ROUTE_CACHE),
        "hits": _CACHE_STATS["hits"],
        "misses": _CACHE_STATS["misses"],
        "sets": _CACHE_STATS["sets"],
        "hit_rate": hit_rate,
        "hit_rate_percentage": f"{hit_rate * 100:.1f}%",
    }


async def clear_cache(pattern: str = ""):
    """Clear cache entries, optionally matching a pattern."""
    if pattern:
        keys_to_remove = [key for key in _ROUTE_CACHE.keys() if pattern in key]
        for key in keys_to_remove:
            del _ROUTE_CACHE[key]
    else:
        _ROUTE_CACHE.clear()

    # Reset stats
    _CACHE_STATS["hits"] = 0
    _CACHE_STATS["misses"] = 0
    _CACHE_STATS["sets"] = 0


def optimize_response(compress: bool = True, cache_control: str = "public, max-age=300"):
    """Decorator for optimizing API responses."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)

            # Add optimization headers if it's a Response object
            if hasattr(result, "headers"):
                if cache_control:
                    result.headers["Cache-Control"] = cache_control

                if compress:
                    result.headers["Content-Encoding"] = "gzip"
                    result.headers["Vary"] = "Accept-Encoding"

                result.headers["X-Optimized"] = "true"

            return result

        return wrapper

    return decorator

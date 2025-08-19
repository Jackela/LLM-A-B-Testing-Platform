"""Singleton container for dependency injection optimization."""

from functools import lru_cache
from typing import Optional

from .container import Container

# Singleton container instance
_container_instance: Optional[Container] = None


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Get singleton container instance for performance optimization.

    This replaces the previous pattern of creating new container instances
    for each request, which was causing performance overhead.

    Returns:
        Container: Singleton container instance
    """
    global _container_instance
    if _container_instance is None:
        _container_instance = Container()
    return _container_instance


def reset_container() -> None:
    """Reset the singleton container (useful for testing).

    This function allows tests to reset the container state
    when needed for isolation between test cases.
    """
    global _container_instance
    _container_instance = None
    # Clear the lru_cache
    get_container.cache_clear()

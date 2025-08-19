"""Performance configuration management for the LLM A/B Testing Platform."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .cache_manager import CacheConfig


@dataclass
class DatabasePerformanceConfig:
    """Database performance configuration."""

    # Connection pool settings
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour
    pool_pre_ping: bool = True

    # Query optimization
    enable_query_cache: bool = True
    query_cache_ttl: int = 300  # 5 minutes
    slow_query_threshold_ms: int = 100

    # Connection optimization
    enable_connection_monitoring: bool = True
    connection_health_check_interval: int = 30


@dataclass
class APIPerformanceConfig:
    """API performance configuration."""

    # Response optimization
    enable_compression: bool = True
    compression_threshold: int = 1024  # Compress responses > 1KB
    compression_level: int = 6

    # Caching
    enable_response_caching: bool = True
    default_cache_ttl: int = 300  # 5 minutes
    cache_vary_headers: list[str] = field(default_factory=lambda: ["Authorization", "Accept"])

    # Request batching
    enable_request_batching: bool = True
    batch_timeout_ms: int = 50
    max_batch_size: int = 100

    # Rate limiting
    default_rate_limit: str = "1000/minute"
    burst_rate_limit: str = "100/second"


@dataclass
class ExternalServiceConfig:
    """External service performance configuration."""

    # HTTP client optimization
    connection_pool_size: int = 100
    connection_pool_maxsize: int = 200
    connection_timeout: int = 10
    read_timeout: int = 30

    # Circuit breaker settings
    failure_threshold: int = 5
    recovery_timeout: int = 60
    expected_exception: tuple = field(default_factory=lambda: (Exception,))

    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 0.5
    retry_statuses: list[int] = field(default_factory=lambda: [429, 502, 503, 504])

    # Batching
    enable_request_batching: bool = True
    batch_size: int = 10
    batch_timeout_ms: int = 100


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""

    # Metrics collection
    enable_metrics: bool = True
    enable_prometheus: bool = True
    prometheus_port: int = 8000
    metrics_export_interval: int = 15  # seconds

    # Alerting
    enable_alerting: bool = True
    alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "response_time_p95_ms": 200.0,
            "error_rate_percent": 1.0,
            "cache_hit_rate_percent": 80.0,
            "memory_usage_percent": 80.0,
            "cpu_usage_percent": 80.0,
            "disk_usage_percent": 90.0,
        }
    )

    # Health checks
    health_check_interval: int = 30
    health_check_timeout: int = 10


@dataclass
class PerformanceConfig:
    """Comprehensive performance configuration."""

    # Component configurations
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    database_config: DatabasePerformanceConfig = field(default_factory=DatabasePerformanceConfig)
    api_config: APIPerformanceConfig = field(default_factory=APIPerformanceConfig)
    external_service_config: ExternalServiceConfig = field(default_factory=ExternalServiceConfig)
    monitoring_config: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Global settings
    environment: str = "development"
    enable_performance_optimization: bool = True
    enable_debug_logging: bool = False

    # Performance targets
    target_response_time_ms: int = 200
    target_throughput_rps: int = 1000
    target_cache_hit_rate: float = 0.8
    target_memory_usage_mb: int = 512
    target_uptime: float = 0.999

    @classmethod
    def from_env(cls) -> "PerformanceConfig":
        """Create configuration from environment variables."""

        # Cache configuration
        cache_config = CacheConfig(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            redis_db=int(os.getenv("REDIS_DB", "1")),  # Use DB 1 for caching
            redis_pool_size=int(os.getenv("REDIS_POOL_SIZE", "20")),
            redis_max_connections=int(os.getenv("REDIS_MAX_CONNECTIONS", "50")),
            memory_max_size=int(os.getenv("MEMORY_CACHE_SIZE", "10000")),
            memory_ttl_default=int(os.getenv("MEMORY_CACHE_TTL", "300")),
            compression_enabled=os.getenv("CACHE_COMPRESSION", "true").lower() == "true",
            enable_metrics=os.getenv("CACHE_METRICS", "true").lower() == "true",
        )

        # Database configuration
        database_config = DatabasePerformanceConfig(
            pool_size=int(os.getenv("DB_POOL_SIZE", "20")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "30")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30")),
            pool_recycle=int(os.getenv("DB_POOL_RECYCLE", "3600")),
            enable_query_cache=os.getenv("DB_QUERY_CACHE", "true").lower() == "true",
            query_cache_ttl=int(os.getenv("DB_QUERY_CACHE_TTL", "300")),
        )

        # API configuration
        api_config = APIPerformanceConfig(
            enable_compression=os.getenv("API_COMPRESSION", "true").lower() == "true",
            compression_threshold=int(os.getenv("API_COMPRESSION_THRESHOLD", "1024")),
            enable_response_caching=os.getenv("API_RESPONSE_CACHE", "true").lower() == "true",
            default_cache_ttl=int(os.getenv("API_CACHE_TTL", "300")),
            enable_request_batching=os.getenv("API_REQUEST_BATCHING", "true").lower() == "true",
        )

        # External service configuration
        external_service_config = ExternalServiceConfig(
            connection_pool_size=int(os.getenv("HTTP_POOL_SIZE", "100")),
            connection_timeout=int(os.getenv("HTTP_CONNECTION_TIMEOUT", "10")),
            read_timeout=int(os.getenv("HTTP_READ_TIMEOUT", "30")),
            failure_threshold=int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
            max_retries=int(os.getenv("HTTP_MAX_RETRIES", "3")),
        )

        # Monitoring configuration
        monitoring_config = MonitoringConfig(
            enable_metrics=os.getenv("ENABLE_METRICS", "true").lower() == "true",
            enable_prometheus=os.getenv("ENABLE_PROMETHEUS", "true").lower() == "true",
            prometheus_port=int(os.getenv("PROMETHEUS_PORT", "8000")),
            enable_alerting=os.getenv("ENABLE_ALERTING", "true").lower() == "true",
        )

        return cls(
            cache_config=cache_config,
            database_config=database_config,
            api_config=api_config,
            external_service_config=external_service_config,
            monitoring_config=monitoring_config,
            environment=os.getenv("ENVIRONMENT", "development"),
            enable_performance_optimization=os.getenv(
                "ENABLE_PERFORMANCE_OPTIMIZATION", "true"
            ).lower()
            == "true",
            enable_debug_logging=os.getenv("DEBUG_PERFORMANCE_LOGGING", "false").lower() == "true",
            target_response_time_ms=int(os.getenv("TARGET_RESPONSE_TIME_MS", "200")),
            target_throughput_rps=int(os.getenv("TARGET_THROUGHPUT_RPS", "1000")),
            target_cache_hit_rate=float(os.getenv("TARGET_CACHE_HIT_RATE", "0.8")),
            target_memory_usage_mb=int(os.getenv("TARGET_MEMORY_USAGE_MB", "512")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment,
            "enable_performance_optimization": self.enable_performance_optimization,
            "enable_debug_logging": self.enable_debug_logging,
            "targets": {
                "response_time_ms": self.target_response_time_ms,
                "throughput_rps": self.target_throughput_rps,
                "cache_hit_rate": self.target_cache_hit_rate,
                "memory_usage_mb": self.target_memory_usage_mb,
                "uptime": self.target_uptime,
            },
            "cache": {
                "redis_url": self.cache_config.redis_url,
                "redis_db": self.cache_config.redis_db,
                "pool_size": self.cache_config.redis_pool_size,
                "memory_max_size": self.cache_config.memory_max_size,
                "compression_enabled": self.cache_config.compression_enabled,
            },
            "database": {
                "pool_size": self.database_config.pool_size,
                "max_overflow": self.database_config.max_overflow,
                "query_cache_enabled": self.database_config.enable_query_cache,
            },
            "api": {
                "compression_enabled": self.api_config.enable_compression,
                "response_caching_enabled": self.api_config.enable_response_caching,
                "request_batching_enabled": self.api_config.enable_request_batching,
            },
            "monitoring": {
                "metrics_enabled": self.monitoring_config.enable_metrics,
                "prometheus_enabled": self.monitoring_config.enable_prometheus,
                "alerting_enabled": self.monitoring_config.enable_alerting,
            },
        }


def get_performance_config() -> PerformanceConfig:
    """Get performance configuration from environment."""
    return PerformanceConfig.from_env()

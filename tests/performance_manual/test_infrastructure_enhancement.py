#!/usr/bin/env python3
"""
Test Infrastructure Enhancement
测试基础设施增强，专注于环境设置、数据管理、监控和执行流程优化
"""

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import aiofiles
import docker
import psutil
import pytest
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TestEnvironmentConfig:
    """测试环境配置"""

    name: str
    description: str

    # 基础设置
    max_concurrent_tests: int = 50
    test_timeout_seconds: int = 300
    retry_attempts: int = 3

    # 资源配置
    memory_limit_mb: int = 2048
    cpu_limit_percent: int = 80
    disk_space_limit_gb: int = 10

    # 网络配置
    api_rate_limit_per_second: int = 100
    connection_pool_size: int = 100
    request_timeout_seconds: int = 30

    # 数据配置
    test_data_cache_size_mb: int = 500
    result_retention_days: int = 30
    enable_test_data_warming: bool = True

    # 监控配置
    enable_real_time_monitoring: bool = True
    metrics_collection_interval: float = 1.0
    performance_alert_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "response_time_p95_ms": 2000,
            "error_rate_percent": 5.0,
            "memory_usage_percent": 85.0,
            "cpu_usage_percent": 80.0,
        }
    )

    # 数据库配置
    database_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "pool_size": 20,
            "max_overflow": 30,
            "pool_timeout": 10,
            "enable_query_optimization": True,
        }
    )

    # 缓存配置
    cache_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "redis_url": "redis://localhost:6379",
            "max_memory": "256mb",
            "eviction_policy": "allkeys-lru",
            "enable_persistence": False,
        }
    )


@dataclass
class TestRunMetrics:
    """测试运行指标"""

    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None

    # 执行统计
    total_samples: int = 0
    processed_samples: int = 0
    successful_samples: int = 0
    failed_samples: int = 0

    # 性能指标
    average_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    throughput_samples_per_second: float = 0.0

    # 资源使用
    peak_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0
    network_io_mb: float = 0.0

    # 成本指标
    total_cost_usd: float = 0.0
    cost_per_sample_usd: float = 0.0

    # 质量指标
    data_quality_score: float = 0.0
    test_coverage_percent: float = 0.0

    # 错误分析
    error_breakdown: Dict[str, int] = field(default_factory=dict)
    warning_count: int = 0


class TestDataManager:
    """测试数据管理器"""

    def __init__(self, data_dir: Path, cache_size_mb: int = 500, enable_compression: bool = True):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.cache_size_mb = cache_size_mb
        self.enable_compression = enable_compression

        # 内存缓存
        self._memory_cache: Dict[str, Any] = {}
        self._cache_access_times: Dict[str, datetime] = {}
        self._current_cache_size = 0

        # 数据统计
        self.data_stats = {
            "datasets_loaded": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "disk_reads": 0,
            "compression_ratio": 0.0,
        }

    async def load_dataset(
        self, dataset_name: str, max_samples: Optional[int] = None, force_reload: bool = False
    ) -> List[Dict[str, Any]]:
        """加载数据集"""

        cache_key = f"{dataset_name}_{max_samples or 'all'}"

        # 检查内存缓存
        if not force_reload and cache_key in self._memory_cache:
            self._cache_access_times[cache_key] = datetime.now()
            self.data_stats["cache_hits"] += 1
            logger.debug(f"📦 从缓存加载数据集: {cache_key}")
            return self._memory_cache[cache_key]

        self.data_stats["cache_misses"] += 1

        # 从磁盘加载
        dataset_file = self.data_dir / f"{dataset_name}.json"
        if not dataset_file.exists():
            raise FileNotFoundError(f"数据集文件不存在: {dataset_file}")

        logger.info(f"💾 从磁盘加载数据集: {dataset_name}")
        start_time = time.time()

        async with aiofiles.open(dataset_file, "r", encoding="utf-8") as f:
            content = await f.read()

        if self.enable_compression:
            # 检查是否为压缩文件
            try:
                import gzip

                if dataset_file.suffix == ".gz":
                    content = gzip.decompress(content.encode()).decode()
            except Exception:
                pass

        dataset = json.loads(content)
        self.data_stats["disk_reads"] += 1

        # 应用样本限制
        if max_samples and len(dataset) > max_samples:
            dataset = dataset[:max_samples]

        load_time = time.time() - start_time
        logger.info(f"✅ 数据集加载完成: {len(dataset)} 样本, 耗时 {load_time:.2f}s")

        # 缓存到内存
        await self._cache_dataset(cache_key, dataset)

        self.data_stats["datasets_loaded"] += 1
        return dataset

    async def _cache_dataset(self, cache_key: str, data: List[Dict[str, Any]]):
        """缓存数据集到内存"""

        # 估算数据大小
        data_size_mb = len(json.dumps(data)) / 1024 / 1024

        # 检查缓存容量
        while self._current_cache_size + data_size_mb > self.cache_size_mb:
            await self._evict_oldest_cache()

        # 添加到缓存
        self._memory_cache[cache_key] = data
        self._cache_access_times[cache_key] = datetime.now()
        self._current_cache_size += data_size_mb

        logger.debug(f"💾 数据集已缓存: {cache_key} ({data_size_mb:.1f}MB)")

    async def _evict_oldest_cache(self):
        """驱逐最旧的缓存"""

        if not self._cache_access_times:
            return

        # 找到最旧的缓存项
        oldest_key = min(self._cache_access_times.keys(), key=lambda k: self._cache_access_times[k])

        # 估算大小并移除
        if oldest_key in self._memory_cache:
            data_size_mb = len(json.dumps(self._memory_cache[oldest_key])) / 1024 / 1024
            del self._memory_cache[oldest_key]
            del self._cache_access_times[oldest_key]
            self._current_cache_size -= data_size_mb

            logger.debug(f"🗑️ 驱逐缓存: {oldest_key} ({data_size_mb:.1f}MB)")

    async def prepare_test_datasets(
        self, required_datasets: List[str], sample_limits: Optional[Dict[str, int]] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """预加载测试所需的数据集"""

        logger.info(f"🔄 预加载 {len(required_datasets)} 个数据集...")

        datasets = {}
        load_tasks = []

        for dataset_name in required_datasets:
            max_samples = sample_limits.get(dataset_name) if sample_limits else None
            task = self.load_dataset(dataset_name, max_samples)
            load_tasks.append((dataset_name, task))

        # 并行加载
        for dataset_name, task in load_tasks:
            try:
                data = await task
                datasets[dataset_name] = data
                logger.info(f"✅ {dataset_name}: {len(data)} 样本")
            except Exception as e:
                logger.error(f"❌ 加载数据集失败 {dataset_name}: {e}")
                datasets[dataset_name] = []

        logger.info(f"🎯 数据集预加载完成: {len(datasets)} 个数据集")
        return datasets

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""

        total_operations = self.data_stats["cache_hits"] + self.data_stats["cache_misses"]
        hit_rate = self.data_stats["cache_hits"] / total_operations if total_operations > 0 else 0

        return {
            "cache_hit_rate": hit_rate,
            "current_cache_size_mb": self._current_cache_size,
            "max_cache_size_mb": self.cache_size_mb,
            "cached_items": len(self._memory_cache),
            "data_stats": self.data_stats.copy(),
        }


class TestEnvironmentManager:
    """测试环境管理器"""

    def __init__(
        self, config: TestEnvironmentConfig, work_dir: Path = Path("tests/performance_workspace")
    ):
        self.config = config
        self.work_dir = work_dir
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # 组件初始化
        self.data_manager = TestDataManager(
            data_dir=self.work_dir / "data", cache_size_mb=config.test_data_cache_size_mb
        )

        # 资源监控
        self._monitoring_active = False
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()
        self._resource_history = []

        # 并发控制
        self._semaphore = asyncio.Semaphore(config.max_concurrent_tests)
        self._executor = ThreadPoolExecutor(max_workers=config.max_concurrent_tests)

        # 测试运行记录
        self.test_runs: List[TestRunMetrics] = []
        self.active_tests: Dict[str, TestRunMetrics] = {}

        # 环境健康状态
        self.health_status = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_latency": 0.0,
            "database_connections": 0,
            "cache_health": "ok",
        }

    async def setup_environment(self) -> bool:
        """设置测试环境"""

        logger.info(f"🚀 设置测试环境: {self.config.name}")

        try:
            # 1. 检查系统资源
            if not await self._check_system_requirements():
                return False

            # 2. 初始化数据库连接池
            await self._setup_database_pool()

            # 3. 初始化缓存
            await self._setup_cache()

            # 4. 启动资源监控
            self._start_resource_monitoring()

            # 5. 预热测试数据
            if self.config.enable_test_data_warming:
                await self._warm_test_data()

            # 6. 验证环境健康状态
            health_check = await self._comprehensive_health_check()

            if health_check["status"] == "healthy":
                logger.info("✅ 测试环境设置完成")
                return True
            else:
                logger.error(f"❌ 环境健康检查失败: {health_check['issues']}")
                return False

        except Exception as e:
            logger.error(f"❌ 环境设置失败: {e}")
            return False

    async def _check_system_requirements(self) -> bool:
        """检查系统资源要求"""

        logger.info("🔍 检查系统资源...")

        # 检查可用内存
        memory = psutil.virtual_memory()
        available_memory_mb = memory.available / 1024 / 1024

        if available_memory_mb < self.config.memory_limit_mb:
            logger.error(f"内存不足: 需要{self.config.memory_limit_mb}MB, 可用{available_memory_mb:.0f}MB")
            return False

        # 检查CPU
        cpu_count = psutil.cpu_count()
        if cpu_count < 2:
            logger.warning("CPU核心数较少，可能影响并发性能")

        # 检查磁盘空间
        disk = psutil.disk_usage(self.work_dir)
        available_gb = disk.free / 1024 / 1024 / 1024

        if available_gb < self.config.disk_space_limit_gb:
            logger.error(f"磁盘空间不足: 需要{self.config.disk_space_limit_gb}GB, 可用{available_gb:.1f}GB")
            return False

        logger.info("✅ 系统资源检查通过")
        return True

    async def _setup_database_pool(self):
        """设置数据库连接池"""

        logger.info("🗄️ 初始化数据库连接池...")

        # 这里可以集成实际的数据库连接池设置
        # 例如: SQLAlchemy, asyncpg 等

        # 模拟数据库池设置
        await asyncio.sleep(0.5)

        self.health_status["database_connections"] = self.config.database_config["pool_size"]
        logger.info("✅ 数据库连接池初始化完成")

    async def _setup_cache(self):
        """设置缓存"""

        logger.info("🔧 初始化缓存系统...")

        # 这里可以集成实际的缓存系统
        # 例如: Redis, Memcached 等

        # 模拟缓存设置
        await asyncio.sleep(0.3)

        self.health_status["cache_health"] = "ok"
        logger.info("✅ 缓存系统初始化完成")

    def _start_resource_monitoring(self):
        """启动资源监控"""

        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._stop_monitoring.clear()

        self._monitor_thread = threading.Thread(target=self._resource_monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        logger.info("📊 资源监控已启动")

    def _resource_monitor_loop(self):
        """资源监控循环"""

        while not self._stop_monitoring.is_set():
            try:
                # 收集系统资源信息
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage(self.work_dir)

                # 更新健康状态
                self.health_status.update(
                    {
                        "cpu_usage": cpu_percent,
                        "memory_usage": memory.percent,
                        "disk_usage": (disk.used / disk.total) * 100,
                    }
                )

                # 记录历史数据
                resource_snapshot = {
                    "timestamp": datetime.now(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_mb": memory.used / 1024 / 1024,
                    "disk_percent": self.health_status["disk_usage"],
                }

                self._resource_history.append(resource_snapshot)

                # 保持历史数据在合理范围内
                if len(self._resource_history) > 3600:  # 1小时的数据
                    self._resource_history = self._resource_history[-1800:]

                # 检查告警阈值
                self._check_resource_alerts(resource_snapshot)

                self._stop_monitoring.wait(self.config.metrics_collection_interval)

            except Exception as e:
                logger.error(f"资源监控错误: {e}")
                self._stop_monitoring.wait(5.0)

    def _check_resource_alerts(self, snapshot: Dict[str, Any]):
        """检查资源告警"""

        thresholds = self.config.performance_alert_thresholds

        # CPU告警
        if snapshot["cpu_percent"] > thresholds.get("cpu_usage_percent", 80):
            logger.warning(f"🚨 CPU使用率过高: {snapshot['cpu_percent']:.1f}%")

        # 内存告警
        if snapshot["memory_percent"] > thresholds.get("memory_usage_percent", 85):
            logger.warning(f"🚨 内存使用率过高: {snapshot['memory_percent']:.1f}%")

        # 磁盘告警
        if snapshot["disk_percent"] > 90:
            logger.warning(f"🚨 磁盘使用率过高: {snapshot['disk_percent']:.1f}%")

    async def _warm_test_data(self):
        """预热测试数据"""

        logger.info("🔥 开始测试数据预热...")

        # 加载常用数据集
        common_datasets = ["arc_easy", "gsm8k"]
        available_datasets = []

        for dataset in common_datasets:
            dataset_file = self.data_manager.data_dir / f"{dataset}.json"
            if dataset_file.exists():
                available_datasets.append(dataset)

        if available_datasets:
            await self.data_manager.prepare_test_datasets(
                available_datasets, sample_limits={dataset: 1000 for dataset in available_datasets}
            )
            logger.info(f"✅ 预热完成: {len(available_datasets)} 个数据集")
        else:
            logger.info("ℹ️ 没有找到可预热的数据集")

    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """综合健康检查"""

        logger.info("🏥 执行综合健康检查...")

        issues = []
        warnings = []

        # 系统资源检查
        if self.health_status["cpu_usage"] > 90:
            issues.append("CPU使用率过高")
        elif self.health_status["cpu_usage"] > 70:
            warnings.append("CPU使用率较高")

        if self.health_status["memory_usage"] > 90:
            issues.append("内存使用率过高")
        elif self.health_status["memory_usage"] > 80:
            warnings.append("内存使用率较高")

        # 缓存健康检查
        cache_stats = self.data_manager.get_cache_stats()
        if cache_stats["cache_hit_rate"] < 0.3:
            warnings.append("缓存命中率较低")

        # 确定整体状态
        if issues:
            status = "unhealthy"
        elif warnings:
            status = "degraded"
        else:
            status = "healthy"

        health_report = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "issues": issues,
            "warnings": warnings,
            "system_health": self.health_status.copy(),
            "cache_stats": cache_stats,
            "environment_config": {
                "name": self.config.name,
                "max_concurrent_tests": self.config.max_concurrent_tests,
                "memory_limit_mb": self.config.memory_limit_mb,
            },
        }

        logger.info(f"🎯 健康检查完成: {status}")
        if warnings:
            logger.warning(f"⚠️ 警告: {', '.join(warnings)}")

        return health_report

    @asynccontextmanager
    async def test_session(self, test_name: str):
        """测试会话上下文管理器"""

        # 创建测试运行记录
        test_metrics = TestRunMetrics(test_name=test_name, start_time=datetime.now())

        self.active_tests[test_name] = test_metrics

        async with self._semaphore:
            try:
                logger.info(f"🚀 开始测试会话: {test_name}")
                yield test_metrics

            except Exception as e:
                logger.error(f"❌ 测试会话失败: {test_name} - {e}")
                test_metrics.failed_samples += 1
                raise

            finally:
                # 完成测试记录
                test_metrics.end_time = datetime.now()

                if test_metrics.total_samples > 0:
                    test_metrics.data_quality_score = (
                        test_metrics.successful_samples / test_metrics.total_samples
                    )

                # 移动到完成列表
                if test_name in self.active_tests:
                    del self.active_tests[test_name]

                self.test_runs.append(test_metrics)

                duration = (test_metrics.end_time - test_metrics.start_time).total_seconds()
                logger.info(f"✅ 测试会话完成: {test_name} (耗时 {duration:.1f}s)")

    async def run_performance_test(
        self, test_func: Callable, test_name: str, test_args: Optional[Dict[str, Any]] = None
    ) -> TestRunMetrics:
        """运行性能测试"""

        async with self.test_session(test_name) as metrics:
            # 准备测试参数
            args = test_args or {}
            args["test_metrics"] = metrics
            args["data_manager"] = self.data_manager
            args["environment"] = self

            # 执行测试
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func(**args)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(self._executor, lambda: test_func(**args))

            # 更新指标
            if isinstance(result, dict):
                if "total_samples" in result:
                    metrics.total_samples = result["total_samples"]
                if "successful_samples" in result:
                    metrics.successful_samples = result["successful_samples"]
                if "total_cost" in result:
                    metrics.total_cost_usd = result["total_cost"]

            return metrics

    def get_environment_status(self) -> Dict[str, Any]:
        """获取环境状态"""

        return {
            "config": {
                "name": self.config.name,
                "max_concurrent_tests": self.config.max_concurrent_tests,
                "memory_limit_mb": self.config.memory_limit_mb,
            },
            "health": self.health_status.copy(),
            "active_tests": len(self.active_tests),
            "completed_tests": len(self.test_runs),
            "cache_stats": self.data_manager.get_cache_stats(),
            "resource_usage": self._get_current_resource_usage(),
            "monitoring_active": self._monitoring_active,
        }

    def _get_current_resource_usage(self) -> Dict[str, float]:
        """获取当前资源使用情况"""

        if self._resource_history:
            latest = self._resource_history[-1]
            return {
                "cpu_percent": latest["cpu_percent"],
                "memory_percent": latest["memory_percent"],
                "memory_mb": latest["memory_mb"],
                "disk_percent": latest["disk_percent"],
            }

        return {"cpu_percent": 0.0, "memory_percent": 0.0, "memory_mb": 0.0, "disk_percent": 0.0}

    async def cleanup_environment(self):
        """清理环境"""

        logger.info("🧹 开始清理测试环境...")

        # 停止监控
        if self._monitoring_active:
            self._monitoring_active = False
            self._stop_monitoring.set()

            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)

        # 关闭线程池
        self._executor.shutdown(wait=True)

        # 清理缓存
        self.data_manager._memory_cache.clear()
        self.data_manager._cache_access_times.clear()

        # 生成最终报告
        final_report = await self._generate_final_report()

        # 保存报告
        report_file = (
            self.work_dir
            / f"test_environment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        async with aiofiles.open(report_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(final_report, ensure_ascii=False, indent=2))

        logger.info(f"📊 环境报告已保存: {report_file}")
        logger.info("✅ 测试环境清理完成")

    async def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""

        total_tests = len(self.test_runs)
        if total_tests == 0:
            return {"message": "没有执行任何测试"}

        # 计算汇总统计
        total_samples = sum(t.total_samples for t in self.test_runs)
        successful_samples = sum(t.successful_samples for t in self.test_runs)
        total_cost = sum(t.total_cost_usd for t in self.test_runs)

        success_rate = successful_samples / total_samples if total_samples > 0 else 0
        avg_cost_per_sample = total_cost / total_samples if total_samples > 0 else 0

        # 性能统计
        response_times = []
        for test in self.test_runs:
            if test.average_response_time_ms > 0:
                response_times.append(test.average_response_time_ms)

        avg_response_time = sum(response_times) / len(response_times) if response_times else 0

        # 资源使用统计
        if self._resource_history:
            cpu_values = [r["cpu_percent"] for r in self._resource_history]
            memory_values = [r["memory_percent"] for r in self._resource_history]

            peak_cpu = max(cpu_values)
            peak_memory = max(memory_values)
            avg_cpu = sum(cpu_values) / len(cpu_values)
            avg_memory = sum(memory_values) / len(memory_values)
        else:
            peak_cpu = peak_memory = avg_cpu = avg_memory = 0

        return {
            "environment_summary": {
                "name": self.config.name,
                "test_duration": (
                    (datetime.now() - min(t.start_time for t in self.test_runs)).total_seconds()
                    if self.test_runs
                    else 0
                ),
                "total_tests_executed": total_tests,
            },
            "performance_summary": {
                "total_samples_processed": total_samples,
                "success_rate": success_rate,
                "average_response_time_ms": avg_response_time,
                "total_cost_usd": total_cost,
                "cost_per_sample_usd": avg_cost_per_sample,
            },
            "resource_utilization": {
                "peak_cpu_percent": peak_cpu,
                "peak_memory_percent": peak_memory,
                "average_cpu_percent": avg_cpu,
                "average_memory_percent": avg_memory,
            },
            "cache_performance": self.data_manager.get_cache_stats(),
            "test_results": [
                {
                    "name": t.test_name,
                    "duration_seconds": (
                        (t.end_time - t.start_time).total_seconds() if t.end_time else 0
                    ),
                    "total_samples": t.total_samples,
                    "success_rate": (
                        t.successful_samples / t.total_samples if t.total_samples > 0 else 0
                    ),
                    "cost_usd": t.total_cost_usd,
                }
                for t in self.test_runs
            ],
            "recommendations": self._generate_environment_recommendations(),
        }

    def _generate_environment_recommendations(self) -> List[str]:
        """生成环境优化建议"""

        recommendations = []

        # 缓存建议
        cache_stats = self.data_manager.get_cache_stats()
        if cache_stats["cache_hit_rate"] < 0.5:
            recommendations.append("缓存命中率较低，考虑增加缓存大小或优化缓存策略")

        # 并发建议
        if len(self.test_runs) > 0:
            avg_test_duration = sum(
                (t.end_time - t.start_time).total_seconds() for t in self.test_runs if t.end_time
            ) / len([t for t in self.test_runs if t.end_time])

            if avg_test_duration > 300:  # 5分钟
                recommendations.append("测试执行时间较长，考虑优化测试数据大小或增加并发数")

        # 资源建议
        if self._resource_history:
            avg_cpu = sum(r["cpu_percent"] for r in self._resource_history) / len(
                self._resource_history
            )
            avg_memory = sum(r["memory_percent"] for r in self._resource_history) / len(
                self._resource_history
            )

            if avg_cpu > 80:
                recommendations.append("CPU使用率较高，考虑优化算法或增加计算资源")
            if avg_memory > 80:
                recommendations.append("内存使用率较高，考虑优化数据结构或增加内存")

        if not recommendations:
            recommendations.append("环境配置合理，性能表现良好")

        return recommendations


# 使用示例和测试
async def example_test_function(
    test_metrics: TestRunMetrics,
    data_manager: TestDataManager,
    environment: TestEnvironmentManager,
    sample_count: int = 100,
) -> Dict[str, Any]:
    """示例测试函数"""

    logger.info(f"🔄 开始示例测试，目标样本数: {sample_count}")

    # 加载测试数据
    try:
        datasets = await data_manager.prepare_test_datasets(
            ["arc_easy"], {"arc_easy": sample_count}
        )

        if "arc_easy" not in datasets or not datasets["arc_easy"]:
            logger.warning("未找到测试数据，使用模拟数据")
            test_data = [{"id": i, "prompt": f"test question {i}"} for i in range(sample_count)]
        else:
            test_data = datasets["arc_easy"][:sample_count]

    except Exception as e:
        logger.warning(f"数据加载失败，使用模拟数据: {e}")
        test_data = [{"id": i, "prompt": f"test question {i}"} for i in range(sample_count)]

    # 更新测试指标
    test_metrics.total_samples = len(test_data)

    # 模拟处理
    successful_count = 0
    total_response_time = 0.0

    for i, sample in enumerate(test_data):
        # 模拟处理时间
        response_time = 0.1 + (i % 10) * 0.05  # 100-600ms
        await asyncio.sleep(response_time)

        # 模拟成功/失败
        if i % 10 != 0:  # 90%成功率
            successful_count += 1
            total_response_time += response_time * 1000  # 转换为ms

        # 更新进度
        if (i + 1) % 20 == 0:
            logger.info(f"📊 处理进度: {i + 1}/{len(test_data)}")

    # 更新测试指标
    test_metrics.processed_samples = len(test_data)
    test_metrics.successful_samples = successful_count
    test_metrics.failed_samples = len(test_data) - successful_count
    test_metrics.average_response_time_ms = (
        total_response_time / successful_count if successful_count > 0 else 0
    )
    test_metrics.total_cost_usd = len(test_data) * 0.001  # 模拟成本

    return {
        "total_samples": len(test_data),
        "successful_samples": successful_count,
        "total_cost": test_metrics.total_cost_usd,
        "average_response_time_ms": test_metrics.average_response_time_ms,
    }


async def test_infrastructure_enhancement():
    """测试基础设施增强演示"""

    # 创建测试配置
    config = TestEnvironmentConfig(
        name="enhanced_performance_test_env",
        description="增强的性能测试环境",
        max_concurrent_tests=20,
        memory_limit_mb=1024,
        enable_test_data_warming=True,
        enable_real_time_monitoring=True,
    )

    # 创建环境管理器
    env_manager = TestEnvironmentManager(config)

    try:
        # 设置环境
        if not await env_manager.setup_environment():
            logger.error("❌ 环境设置失败")
            return

        # 运行测试
        logger.info("🚀 开始运行性能测试...")

        test_tasks = []
        for i in range(3):
            test_name = f"enhanced_test_{i + 1}"
            task = env_manager.run_performance_test(
                example_test_function, test_name, {"sample_count": 50}
            )
            test_tasks.append(task)

        # 等待所有测试完成
        results = await asyncio.gather(*test_tasks)

        # 显示结果
        logger.info("📊 测试结果汇总:")
        for result in results:
            logger.info(
                f"  {result.test_name}: "
                f"{result.successful_samples}/{result.total_samples} 成功 "
                f"(成本: ${result.total_cost_usd:.4f})"
            )

        # 显示环境状态
        status = env_manager.get_environment_status()
        logger.info(
            f"🏥 环境状态: {status['health']['cpu_usage']:.1f}% CPU, {status['health']['memory_usage']:.1f}% 内存"
        )
        logger.info(f"📦 缓存效率: {status['cache_stats']['cache_hit_rate']:.1%} 命中率")

    finally:
        await env_manager.cleanup_environment()


if __name__ == "__main__":
    asyncio.run(test_infrastructure_enhancement())

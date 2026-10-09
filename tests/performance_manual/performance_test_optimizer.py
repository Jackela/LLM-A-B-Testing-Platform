#!/usr/bin/env python3
"""
Performance Test Infrastructure Optimizer
专为LLM A/B测试平台设计的性能测试优化器
"""

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiofiles
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""

    test_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    samples_processed: int = 0
    total_samples: int = 0
    success_rate: float = 0.0
    average_response_time: float = 0.0
    throughput: float = 0.0  # samples per second

    # 资源使用指标
    peak_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0
    network_io_mb: float = 0.0

    # 成本相关指标
    total_cost: float = 0.0
    cost_per_sample: float = 0.0

    # 错误统计
    error_count: int = 0
    error_types: Dict[str, int] = field(default_factory=dict)

    # 缓存效率
    cache_hit_rate: float = 0.0
    cache_size_mb: float = 0.0


class PerformanceTestOptimizer:
    """性能测试优化器核心类"""

    def __init__(
        self,
        cache_dir: Path = Path("tests/performance_cache"),
        max_concurrent_tasks: int = 50,
        batch_size: int = 10,
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_concurrent_tasks = max_concurrent_tasks
        self.batch_size = batch_size

        # 性能监控组件
        self._metrics_cache: Dict[str, Any] = {}
        self._resource_monitor = None
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()

        # 数据缓存组件
        self._dataset_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._result_cache: Dict[str, Any] = {}

        # 并发控制
        self._semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)

        # 性能统计
        self.metrics = PerformanceMetrics(test_name="optimizer", start_time=datetime.now())

    def start_monitoring(self):
        """启动性能监控"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return

        self._stop_monitoring.clear()
        self._monitoring_thread = threading.Thread(target=self._monitor_resources)
        self._monitoring_thread.daemon = True
        self._monitoring_thread.start()
        logger.info("🔍 性能监控已启动")

    def stop_monitoring(self):
        """停止性能监控"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=1.0)
        logger.info("📊 性能监控已停止")

    def _monitor_resources(self):
        """资源监控线程"""
        memory_samples = []
        cpu_samples = []

        while not self._stop_monitoring.is_set():
            try:
                # 内存使用
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                memory_samples.append(memory_mb)

                # CPU使用
                cpu_percent = psutil.cpu_percent(interval=None)
                cpu_samples.append(cpu_percent)

                # 更新峰值
                self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)

                # 每10秒更新平均值
                if len(cpu_samples) >= 10:
                    self.metrics.average_cpu_percent = sum(cpu_samples) / len(cpu_samples)
                    cpu_samples = cpu_samples[-5:]  # 保留最近5个样本

                time.sleep(1)

            except Exception as e:
                logger.warning(f"资源监控错误: {e}")
                time.sleep(1)

    async def load_dataset_cached(
        self, dataset_path: Path, max_samples: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """缓存优化的数据集加载"""

        cache_key = f"{dataset_path.name}_{max_samples or 'all'}"

        # 检查内存缓存
        if cache_key in self._dataset_cache:
            logger.info(f"📦 从内存缓存加载数据集: {cache_key}")
            return self._dataset_cache[cache_key]

        # 检查磁盘缓存
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                async with aiofiles.open(cache_file, "r", encoding="utf-8") as f:
                    content = await f.read()
                    cached_data = json.loads(content)
                    self._dataset_cache[cache_key] = cached_data
                    logger.info(f"💾 从磁盘缓存加载数据集: {cache_key}")
                    return cached_data
            except Exception as e:
                logger.warning(f"磁盘缓存加载失败: {e}")

        # 加载原始数据
        logger.info(f"🔄 加载原始数据集: {dataset_path}")
        start_time = time.time()

        async with aiofiles.open(dataset_path, "r", encoding="utf-8") as f:
            content = await f.read()
            dataset = json.loads(content)

        if max_samples:
            dataset = dataset[:max_samples]

        load_time = time.time() - start_time
        logger.info(f"✅ 数据集加载完成: {len(dataset)} 样本, 耗时 {load_time:.2f}s")

        # 缓存到内存和磁盘
        self._dataset_cache[cache_key] = dataset

        try:
            async with aiofiles.open(cache_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(dataset, ensure_ascii=False, indent=2))
        except Exception as e:
            logger.warning(f"磁盘缓存保存失败: {e}")

        return dataset

    async def batch_process(
        self,
        items: List[Any],
        process_func: Callable,
        batch_size: Optional[int] = None,
        progress_callback: Optional[Callable] = None,
    ) -> List[Any]:
        """批量并发处理优化器"""

        if not items:
            return []

        batch_size = batch_size or self.batch_size
        total_items = len(items)
        results = []

        # 分批处理
        for batch_idx in range(0, total_items, batch_size):
            batch = items[batch_idx : batch_idx + batch_size]

            # 并发处理批次
            batch_tasks = []
            for item in batch:
                task = self._process_with_semaphore(process_func, item)
                batch_tasks.append(task)

            # 等待批次完成
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            # 处理结果
            valid_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    self.metrics.error_count += 1
                    error_type = type(result).__name__
                    self.metrics.error_types[error_type] = (
                        self.metrics.error_types.get(error_type, 0) + 1
                    )
                    logger.error(f"批次处理错误: {result}")
                else:
                    valid_results.append(result)

            results.extend(valid_results)

            # 进度回调
            if progress_callback:
                progress = min(batch_idx + batch_size, total_items) / total_items
                await progress_callback(progress, len(results), self.metrics)

            # 批次间延迟，避免过载
            if batch_idx + batch_size < total_items:
                await asyncio.sleep(0.05)

        return results

    async def _process_with_semaphore(self, process_func: Callable, item: Any) -> Any:
        """使用信号量控制并发的处理函数"""
        async with self._semaphore:
            if asyncio.iscoroutinefunction(process_func):
                return await process_func(item)
            else:
                # 在线程池中执行同步函数
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(self._executor, process_func, item)

    def cache_result(self, key: str, result: Any, ttl_hours: int = 24):
        """缓存测试结果"""
        cache_entry = {
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "ttl_hours": ttl_hours,
        }

        # 内存缓存
        self._result_cache[key] = cache_entry

        # 磁盘缓存
        cache_file = self.cache_dir / f"result_{key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"结果缓存保存失败: {e}")

    def get_cached_result(self, key: str) -> Optional[Any]:
        """获取缓存的测试结果"""

        # 检查内存缓存
        if key in self._result_cache:
            entry = self._result_cache[key]
            timestamp = datetime.fromisoformat(entry["timestamp"])
            ttl = timedelta(hours=entry["ttl_hours"])

            if datetime.now() - timestamp < ttl:
                logger.info(f"📦 使用内存缓存结果: {key}")
                return entry["result"]
            else:
                # 过期，删除
                del self._result_cache[key]

        # 检查磁盘缓存
        cache_file = self.cache_dir / f"result_{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                    timestamp = datetime.fromisoformat(entry["timestamp"])
                    ttl = timedelta(hours=entry["ttl_hours"])

                    if datetime.now() - timestamp < ttl:
                        # 加载到内存缓存
                        self._result_cache[key] = entry
                        logger.info(f"💾 使用磁盘缓存结果: {key}")
                        return entry["result"]
                    else:
                        # 过期，删除文件
                        cache_file.unlink()
            except Exception as e:
                logger.warning(f"磁盘缓存读取失败: {e}")

        return None

    def update_metrics(
        self, samples_processed: int = 0, success_count: int = 0, total_cost: float = 0.0
    ):
        """更新性能指标"""

        self.metrics.samples_processed += samples_processed
        if samples_processed > 0:
            self.metrics.success_rate = success_count / samples_processed

        self.metrics.total_cost += total_cost
        if self.metrics.samples_processed > 0:
            self.metrics.cost_per_sample = self.metrics.total_cost / self.metrics.samples_processed

        # 计算吞吐量
        if self.metrics.start_time:
            elapsed = (datetime.now() - self.metrics.start_time).total_seconds()
            if elapsed > 0:
                self.metrics.throughput = self.metrics.samples_processed / elapsed

    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能优化报告"""

        self.metrics.end_time = datetime.now()
        total_time = (self.metrics.end_time - self.metrics.start_time).total_seconds()

        report = {
            "performance_optimization_report": {
                "timestamp": self.metrics.end_time.isoformat(),
                "test_duration_seconds": total_time,
                "optimization_metrics": {
                    "total_samples_processed": self.metrics.samples_processed,
                    "success_rate": round(self.metrics.success_rate, 3),
                    "throughput_samples_per_second": round(self.metrics.throughput, 2),
                    "average_response_time_ms": round(self.metrics.average_response_time, 2),
                },
                "resource_usage": {
                    "peak_memory_mb": round(self.metrics.peak_memory_mb, 2),
                    "average_cpu_percent": round(self.metrics.average_cpu_percent, 1),
                    "network_io_mb": round(self.metrics.network_io_mb, 2),
                },
                "cost_analysis": {
                    "total_cost_usd": round(self.metrics.total_cost, 6),
                    "cost_per_sample_usd": round(self.metrics.cost_per_sample, 6),
                },
                "error_analysis": {
                    "total_errors": self.metrics.error_count,
                    "error_types": self.metrics.error_types,
                    "error_rate": round(
                        self.metrics.error_count / max(self.metrics.samples_processed, 1), 3
                    ),
                },
                "cache_performance": {
                    "hit_rate": round(self.metrics.cache_hit_rate, 3),
                    "cache_size_mb": round(self.metrics.cache_size_mb, 2),
                    "cached_datasets": len(self._dataset_cache),
                    "cached_results": len(self._result_cache),
                },
                "optimization_recommendations": self._generate_recommendations(),
            }
        }

        return report

    def _generate_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        # 吞吐量建议
        if self.metrics.throughput < 5.0:
            recommendations.append("考虑增加并发数或优化处理逻辑以提高吞吐量")

        # 成功率建议
        if self.metrics.success_rate < 0.95:
            recommendations.append("成功率较低，检查错误处理和重试机制")

        # 内存使用建议
        if self.metrics.peak_memory_mb > 1000:
            recommendations.append("内存使用较高，考虑优化数据结构或增加分页处理")

        # CPU使用建议
        if self.metrics.average_cpu_percent > 80:
            recommendations.append("CPU使用率高，考虑优化算法或减少并发数")

        # 成本建议
        if self.metrics.cost_per_sample > 0.01:
            recommendations.append("每样本成本较高，考虑使用更便宜的模型或优化提示")

        # 缓存建议
        if self.metrics.cache_hit_rate < 0.5:
            recommendations.append("缓存命中率低，考虑调整缓存策略或增加缓存大小")

        # 错误率建议
        error_rate = self.metrics.error_count / max(self.metrics.samples_processed, 1)
        if error_rate > 0.05:
            recommendations.append("错误率超过5%，需要改进错误处理机制")

        if not recommendations:
            recommendations.append("性能表现良好，继续保持当前优化策略")

        return recommendations

    async def cleanup(self):
        """清理资源"""
        self.stop_monitoring()
        self._executor.shutdown(wait=True)

        # 清理过期缓存
        await self._cleanup_expired_cache()

        logger.info("🧹 性能优化器资源清理完成")

    async def _cleanup_expired_cache(self):
        """清理过期缓存文件"""
        try:
            cache_files = list(self.cache_dir.glob("*.json"))
            cleaned_count = 0

            for cache_file in cache_files:
                try:
                    with open(cache_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "timestamp" in data and "ttl_hours" in data:
                            timestamp = datetime.fromisoformat(data["timestamp"])
                            ttl = timedelta(hours=data["ttl_hours"])

                            if datetime.now() - timestamp > ttl:
                                cache_file.unlink()
                                cleaned_count += 1
                except Exception:
                    # 删除损坏的缓存文件
                    cache_file.unlink()
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(f"🗑️ 清理了 {cleaned_count} 个过期缓存文件")

        except Exception as e:
            logger.warning(f"缓存清理失败: {e}")


# 使用示例和测试
async def example_usage():
    """性能优化器使用示例"""

    optimizer = PerformanceTestOptimizer(
        cache_dir=Path("tests/performance_cache"), max_concurrent_tasks=20, batch_size=5
    )

    try:
        # 启动监控
        optimizer.start_monitoring()

        # 模拟数据加载
        logger.info("📚 开始数据加载测试...")
        datasets_dir = Path("data/processed")

        if (datasets_dir / "arc_easy.json").exists():
            dataset = await optimizer.load_dataset_cached(
                datasets_dir / "arc_easy.json", max_samples=100
            )
            logger.info(f"✅ 加载数据集: {len(dataset)} 样本")

        # 模拟批量处理
        async def mock_process_item(item):
            await asyncio.sleep(0.1)  # 模拟处理时间
            return {"processed": True, "item_id": item.get("id", "unknown")}

        async def progress_callback(progress, completed, metrics):
            logger.info(f"📊 处理进度: {progress:.1%} ({completed} 完成)")

        if "dataset" in locals():
            logger.info("🔄 开始批量处理测试...")
            results = await optimizer.batch_process(
                dataset[:20],  # 测试20个样本
                mock_process_item,
                batch_size=5,
                progress_callback=progress_callback,
            )

            optimizer.update_metrics(
                samples_processed=len(results),
                success_count=len(results),
                total_cost=0.05,  # 模拟成本
            )

            logger.info(f"✅ 处理完成: {len(results)} 结果")

        # 生成性能报告
        report = optimizer.generate_performance_report()

        # 保存报告
        report_file = Path("performance_optimization_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"📊 性能报告已保存: {report_file}")
        logger.info("🎯 优化建议:")
        for recommendation in report["performance_optimization_report"][
            "optimization_recommendations"
        ]:
            logger.info(f"  • {recommendation}")

    finally:
        await optimizer.cleanup()


if __name__ == "__main__":
    asyncio.run(example_usage())

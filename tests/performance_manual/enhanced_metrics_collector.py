#!/usr/bin/env python3
"""
Enhanced Performance Metrics Collector
增强的性能指标收集器，专为LLM A/B测试平台优化
"""

import asyncio
import json
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """性能快照数据类"""

    timestamp: datetime

    # 响应时间指标
    response_time_ms: float
    request_id: str
    endpoint: str
    method: str
    status_code: int

    # 系统资源指标
    cpu_percent: float
    memory_mb: float
    memory_percent: float

    # 网络指标
    request_size_bytes: int = 0
    response_size_bytes: int = 0

    # 应用特定指标
    provider: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None

    # 缓存指标
    cache_hit: Optional[bool] = None
    cache_key: Optional[str] = None

    # 自定义指标
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBaseline:
    """性能基准数据类"""

    name: str
    created_at: datetime

    # 响应时间基准
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

    # 吞吐量基准
    max_throughput_rps: float
    sustainable_throughput_rps: float

    # 资源使用基准
    baseline_cpu_percent: float
    baseline_memory_mb: float

    # 成本基准
    avg_cost_per_request: float
    cost_efficiency_score: float

    # 质量基准
    error_rate_threshold: float
    success_rate_target: float

    # 元数据
    test_conditions: Dict[str, Any] = field(default_factory=dict)
    validation_data: Dict[str, Any] = field(default_factory=dict)


class EnhancedMetricsCollector:
    """增强的性能指标收集器"""

    def __init__(
        self,
        collection_interval: float = 1.0,
        max_snapshots: int = 10000,
        baseline_file: Optional[Path] = None,
    ):

        self.collection_interval = collection_interval
        self.max_snapshots = max_snapshots
        self.baseline_file = baseline_file or Path("performance_baselines.json")

        # 数据存储
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.baselines: Dict[str, PerformanceBaseline] = {}

        # 实时统计
        self.current_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0.0,
            "total_cost": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        # 滑动窗口统计
        self.window_size = 100
        self.response_times_window = deque(maxlen=self.window_size)
        self.throughput_window = deque(maxlen=60)  # 1分钟窗口

        # 监控控制
        self._monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()

        # 回调函数
        self.alert_callbacks: List[Callable] = []
        self.real_time_callbacks: List[Callable] = []

        # 性能阈值
        self.thresholds = {
            "response_time_p95_ms": 2000,
            "error_rate_percent": 5.0,
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "cost_per_request": 0.01,
        }

        # 加载现有基准
        self._load_baselines()

    def start_monitoring(self):
        """启动性能监控"""
        if self._monitoring:
            return

        self._monitoring = True
        self._stop_event.clear()

        self._monitor_thread = threading.Thread(target=self._monitor_system_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        logger.info("🔍 增强性能监控已启动")

    def stop_monitoring(self):
        """停止性能监控"""
        if not self._monitoring:
            return

        self._monitoring = False
        self._stop_event.set()

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        logger.info("📊 增强性能监控已停止")

    def _monitor_system_resources(self):
        """系统资源监控线程"""
        while not self._stop_event.is_set():
            try:
                # 收集系统指标
                cpu_percent = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()

                # 更新滑动窗口
                current_time = datetime.now()
                requests_in_window = len(
                    [s for s in self.snapshots if (current_time - s.timestamp).total_seconds() < 60]
                )
                self.throughput_window.append(requests_in_window)

                # 检查阈值并触发告警
                self._check_thresholds(cpu_percent, memory.percent)

                # 调用实时回调
                for callback in self.real_time_callbacks:
                    try:
                        callback(
                            {
                                "timestamp": current_time,
                                "cpu_percent": cpu_percent,
                                "memory_percent": memory.percent,
                                "throughput_rps": (
                                    statistics.mean(self.throughput_window)
                                    if self.throughput_window
                                    else 0
                                ),
                                "avg_response_time": self._get_current_avg_response_time(),
                            }
                        )
                    except Exception as e:
                        logger.warning(f"实时回调执行失败: {e}")

                self._stop_event.wait(self.collection_interval)

            except Exception as e:
                logger.error(f"系统资源监控错误: {e}")
                self._stop_event.wait(self.collection_interval)

    def record_request(
        self,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        request_size: int = 0,
        response_size: int = 0,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
        cache_hit: Optional[bool] = None,
        cache_key: Optional[str] = None,
        custom_metrics: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> PerformanceSnapshot:
        """记录单个请求的性能数据"""

        # 收集系统指标
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()

        # 创建性能快照
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            response_time_ms=response_time_ms,
            request_id=request_id or f"req_{int(time.time() * 1000)}",
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            cpu_percent=cpu_percent,
            memory_mb=memory.used / 1024 / 1024,
            memory_percent=memory.percent,
            request_size_bytes=request_size,
            response_size_bytes=response_size,
            provider=provider,
            model=model,
            tokens_used=tokens_used,
            cost=cost,
            cache_hit=cache_hit,
            cache_key=cache_key,
            custom_metrics=custom_metrics or {},
        )

        # 存储快照
        self.snapshots.append(snapshot)

        # 更新实时统计
        self._update_real_time_stats(snapshot)

        # 更新滑动窗口
        self.response_times_window.append(response_time_ms)

        return snapshot

    def _update_real_time_stats(self, snapshot: PerformanceSnapshot):
        """更新实时统计数据"""
        self.current_stats["total_requests"] += 1

        if 200 <= snapshot.status_code < 400:
            self.current_stats["successful_requests"] += 1
        else:
            self.current_stats["failed_requests"] += 1

        self.current_stats["total_response_time"] += snapshot.response_time_ms

        if snapshot.cost:
            self.current_stats["total_cost"] += snapshot.cost

        if snapshot.cache_hit is not None:
            if snapshot.cache_hit:
                self.current_stats["cache_hits"] += 1
            else:
                self.current_stats["cache_misses"] += 1

    def _check_thresholds(self, cpu_percent: float, memory_percent: float):
        """检查性能阈值并触发告警"""
        alerts = []

        # CPU告警
        if cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(
                {
                    "type": "cpu_high",
                    "value": cpu_percent,
                    "threshold": self.thresholds["cpu_percent"],
                    "severity": "warning",
                }
            )

        # 内存告警
        if memory_percent > self.thresholds["memory_percent"]:
            alerts.append(
                {
                    "type": "memory_high",
                    "value": memory_percent,
                    "threshold": self.thresholds["memory_percent"],
                    "severity": "warning",
                }
            )

        # 响应时间告警
        if len(self.response_times_window) >= 10:
            p95_time = np.percentile(list(self.response_times_window), 95)
            if p95_time > self.thresholds["response_time_p95_ms"]:
                alerts.append(
                    {
                        "type": "response_time_high",
                        "value": p95_time,
                        "threshold": self.thresholds["response_time_p95_ms"],
                        "severity": "critical",
                    }
                )

        # 错误率告警
        if self.current_stats["total_requests"] >= 10:
            error_rate = (
                self.current_stats["failed_requests"] / self.current_stats["total_requests"]
            ) * 100
            if error_rate > self.thresholds["error_rate_percent"]:
                alerts.append(
                    {
                        "type": "error_rate_high",
                        "value": error_rate,
                        "threshold": self.thresholds["error_rate_percent"],
                        "severity": "critical",
                    }
                )

        # 触发告警回调
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.warning(f"告警回调执行失败: {e}")

    def _get_current_avg_response_time(self) -> float:
        """获取当前平均响应时间"""
        if not self.response_times_window:
            return 0.0
        return statistics.mean(self.response_times_window)

    def create_baseline(
        self,
        name: str,
        test_duration_minutes: int = 10,
        target_rps: int = 100,
        test_conditions: Optional[Dict[str, Any]] = None,
    ) -> PerformanceBaseline:
        """创建性能基准"""

        logger.info(f"🎯 开始创建性能基准: {name}")

        # 清空现有数据
        initial_snapshots = len(self.snapshots)

        # 等待收集足够数据
        end_time = datetime.now() + timedelta(minutes=test_duration_minutes)

        while datetime.now() < end_time:
            await asyncio.sleep(1)

            if len(self.snapshots) - initial_snapshots >= target_rps * test_duration_minutes:
                break

        # 分析收集的数据
        recent_snapshots = list(self.snapshots)[initial_snapshots:]

        if not recent_snapshots:
            raise ValueError("没有收集到足够的性能数据来创建基准")

        # 计算基准指标
        response_times = [
            s.response_time_ms for s in recent_snapshots if 200 <= s.status_code < 400
        ]

        if not response_times:
            raise ValueError("没有成功的请求数据来创建基准")

        avg_response_time = statistics.mean(response_times)
        p50_response_time = np.percentile(response_times, 50)
        p95_response_time = np.percentile(response_times, 95)
        p99_response_time = np.percentile(response_times, 99)

        # 计算吞吐量
        duration_seconds = test_duration_minutes * 60
        max_throughput = len(recent_snapshots) / duration_seconds
        sustainable_throughput = len(response_times) / duration_seconds

        # 计算资源使用基准
        cpu_values = [s.cpu_percent for s in recent_snapshots]
        memory_values = [s.memory_mb for s in recent_snapshots]

        baseline_cpu = statistics.mean(cpu_values)
        baseline_memory = statistics.mean(memory_values)

        # 计算成本基准
        costs = [s.cost for s in recent_snapshots if s.cost is not None]
        avg_cost = statistics.mean(costs) if costs else 0.0

        # 计算成本效率得分 (响应时间 vs 成本)
        cost_efficiency = (
            (1000 / avg_response_time) / max(avg_cost, 0.001) if avg_cost > 0 else 1000
        )

        # 计算质量指标
        total_requests = len(recent_snapshots)
        successful_requests = len(response_times)
        error_rate = ((total_requests - successful_requests) / total_requests) * 100
        success_rate = (successful_requests / total_requests) * 100

        # 创建基准对象
        baseline = PerformanceBaseline(
            name=name,
            created_at=datetime.now(),
            avg_response_time_ms=avg_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            max_throughput_rps=max_throughput,
            sustainable_throughput_rps=sustainable_throughput,
            baseline_cpu_percent=baseline_cpu,
            baseline_memory_mb=baseline_memory,
            avg_cost_per_request=avg_cost,
            cost_efficiency_score=cost_efficiency,
            error_rate_threshold=max(error_rate, 1.0),  # 至少1%的阈值
            success_rate_target=min(success_rate, 99.0),  # 最多99%的目标
            test_conditions=test_conditions or {},
            validation_data={
                "total_requests": total_requests,
                "test_duration_minutes": test_duration_minutes,
                "data_points": len(recent_snapshots),
            },
        )

        # 保存基准
        self.baselines[name] = baseline
        self._save_baselines()

        logger.info(f"✅ 性能基准创建完成: {name}")
        logger.info(f"   平均响应时间: {avg_response_time:.1f}ms")
        logger.info(f"   P95响应时间: {p95_response_time:.1f}ms")
        logger.info(f"   可持续吞吐量: {sustainable_throughput:.1f} RPS")
        logger.info(f"   成功率: {success_rate:.1f}%")

        return baseline

    def compare_to_baseline(
        self, baseline_name: str, analysis_window_minutes: int = 5
    ) -> Dict[str, Any]:
        """与基准进行对比分析"""

        if baseline_name not in self.baselines:
            raise ValueError(f"基准 '{baseline_name}' 不存在")

        baseline = self.baselines[baseline_name]

        # 获取最近的性能数据
        cutoff_time = datetime.now() - timedelta(minutes=analysis_window_minutes)
        recent_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]

        if not recent_snapshots:
            raise ValueError("没有足够的最近性能数据进行对比")

        # 计算当前指标
        successful_snapshots = [s for s in recent_snapshots if 200 <= s.status_code < 400]
        response_times = [s.response_time_ms for s in successful_snapshots]

        if not response_times:
            raise ValueError("没有成功的请求数据进行对比")

        current_avg_response = statistics.mean(response_times)
        current_p95_response = np.percentile(response_times, 95)
        current_p99_response = np.percentile(response_times, 99)

        duration_seconds = analysis_window_minutes * 60
        current_throughput = len(successful_snapshots) / duration_seconds

        # 计算对比结果
        comparison = {
            "baseline_name": baseline_name,
            "analysis_period_minutes": analysis_window_minutes,
            "timestamp": datetime.now().isoformat(),
            "response_time_comparison": {
                "avg_response_time": {
                    "current": current_avg_response,
                    "baseline": baseline.avg_response_time_ms,
                    "change_percent": (
                        (current_avg_response - baseline.avg_response_time_ms)
                        / baseline.avg_response_time_ms
                    )
                    * 100,
                    "status": (
                        "improved"
                        if current_avg_response < baseline.avg_response_time_ms
                        else "degraded"
                    ),
                },
                "p95_response_time": {
                    "current": current_p95_response,
                    "baseline": baseline.p95_response_time_ms,
                    "change_percent": (
                        (current_p95_response - baseline.p95_response_time_ms)
                        / baseline.p95_response_time_ms
                    )
                    * 100,
                    "status": (
                        "improved"
                        if current_p95_response < baseline.p95_response_time_ms
                        else "degraded"
                    ),
                },
            },
            "throughput_comparison": {
                "current_rps": current_throughput,
                "baseline_rps": baseline.sustainable_throughput_rps,
                "change_percent": (
                    (current_throughput - baseline.sustainable_throughput_rps)
                    / baseline.sustainable_throughput_rps
                )
                * 100,
                "status": (
                    "improved"
                    if current_throughput > baseline.sustainable_throughput_rps
                    else "degraded"
                ),
            },
            "overall_assessment": self._assess_performance_change(
                current_avg_response,
                baseline.avg_response_time_ms,
                current_throughput,
                baseline.sustainable_throughput_rps,
            ),
        }

        return comparison

    def _assess_performance_change(
        self,
        current_response_time: float,
        baseline_response_time: float,
        current_throughput: float,
        baseline_throughput: float,
    ) -> Dict[str, Any]:
        """评估整体性能变化"""

        response_change = (
            (current_response_time - baseline_response_time) / baseline_response_time
        ) * 100
        throughput_change = ((current_throughput - baseline_throughput) / baseline_throughput) * 100

        # 计算综合得分
        performance_score = 0

        # 响应时间权重50%
        if response_change <= -10:  # 改善超过10%
            performance_score += 50
        elif response_change <= 0:  # 有改善
            performance_score += 25
        elif response_change <= 10:  # 轻微退化
            performance_score += 10
        # 退化超过10%得0分

        # 吞吐量权重50%
        if throughput_change >= 10:  # 改善超过10%
            performance_score += 50
        elif throughput_change >= 0:  # 有改善
            performance_score += 25
        elif throughput_change >= -10:  # 轻微退化
            performance_score += 10
        # 退化超过10%得0分

        # 确定整体状态
        if performance_score >= 75:
            status = "significantly_improved"
        elif performance_score >= 50:
            status = "improved"
        elif performance_score >= 25:
            status = "stable"
        else:
            status = "degraded"

        return {
            "performance_score": performance_score,
            "status": status,
            "response_time_change_percent": round(response_change, 2),
            "throughput_change_percent": round(throughput_change, 2),
            "recommendation": self._get_performance_recommendation(
                status, response_change, throughput_change
            ),
        }

    def _get_performance_recommendation(
        self, status: str, response_change: float, throughput_change: float
    ) -> str:
        """根据性能变化生成建议"""

        if status == "significantly_improved":
            return "性能显著改善，可考虑进一步优化或增加负载"
        elif status == "improved":
            return "性能有所改善，继续监控以确保稳定性"
        elif status == "stable":
            return "性能基本稳定，注意监控资源使用情况"
        else:
            recommendations = ["性能出现退化，建议："]

            if response_change > 10:
                recommendations.append("- 检查响应时间退化的原因")
                recommendations.append("- 优化慢查询或API调用")

            if throughput_change < -10:
                recommendations.append("- 检查系统瓶颈")
                recommendations.append("- 考虑增加资源或优化并发处理")

            return " ".join(recommendations)

    def _save_baselines(self):
        """保存基准到文件"""
        try:
            baselines_data = {}
            for name, baseline in self.baselines.items():
                baselines_data[name] = asdict(baseline)
                # 转换datetime为ISO格式
                baselines_data[name]["created_at"] = baseline.created_at.isoformat()

            with open(self.baseline_file, "w", encoding="utf-8") as f:
                json.dump(baselines_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"保存基准失败: {e}")

    def _load_baselines(self):
        """从文件加载基准"""
        try:
            if self.baseline_file.exists():
                with open(self.baseline_file, "r", encoding="utf-8") as f:
                    baselines_data = json.load(f)

                for name, data in baselines_data.items():
                    # 转换ISO格式为datetime
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                    self.baselines[name] = PerformanceBaseline(**data)

                logger.info(f"✅ 加载了 {len(self.baselines)} 个性能基准")

        except Exception as e:
            logger.warning(f"加载基准失败: {e}")

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加告警回调函数"""
        self.alert_callbacks.append(callback)

    def add_real_time_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """添加实时监控回调函数"""
        self.real_time_callbacks.append(callback)

    def generate_performance_report(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """生成性能报告"""

        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
        relevant_snapshots = [s for s in self.snapshots if s.timestamp > cutoff_time]

        if not relevant_snapshots:
            return {"error": "没有足够的数据生成报告"}

        # 基础统计
        total_requests = len(relevant_snapshots)
        successful_requests = [s for s in relevant_snapshots if 200 <= s.status_code < 400]
        failed_requests = [s for s in relevant_snapshots if not (200 <= s.status_code < 400)]

        response_times = [s.response_time_ms for s in successful_requests]

        # 计算统计指标
        report = {
            "report_period": {
                "hours": time_range_hours,
                "start_time": cutoff_time.isoformat(),
                "end_time": datetime.now().isoformat(),
            },
            "request_statistics": {
                "total_requests": total_requests,
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": (
                    len(successful_requests) / total_requests if total_requests > 0 else 0
                ),
                "error_rate": len(failed_requests) / total_requests if total_requests > 0 else 0,
            },
            "response_time_analysis": {},
            "throughput_analysis": {},
            "resource_usage_analysis": {},
            "cost_analysis": {},
            "cache_analysis": {},
            "baseline_comparisons": {},
            "recommendations": [],
        }

        # 响应时间分析
        if response_times:
            report["response_time_analysis"] = {
                "average_ms": statistics.mean(response_times),
                "median_ms": statistics.median(response_times),
                "p95_ms": np.percentile(response_times, 95),
                "p99_ms": np.percentile(response_times, 99),
                "min_ms": min(response_times),
                "max_ms": max(response_times),
                "std_dev_ms": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            }

        # 吞吐量分析
        if time_range_hours > 0:
            rps = total_requests / (time_range_hours * 3600)
            successful_rps = len(successful_requests) / (time_range_hours * 3600)

            report["throughput_analysis"] = {
                "total_rps": rps,
                "successful_rps": successful_rps,
                "peak_hour_rps": self._calculate_peak_hour_rps(relevant_snapshots),
            }

        # 资源使用分析
        cpu_values = [s.cpu_percent for s in relevant_snapshots]
        memory_values = [s.memory_mb for s in relevant_snapshots]

        if cpu_values:
            report["resource_usage_analysis"] = {
                "cpu_usage": {
                    "average_percent": statistics.mean(cpu_values),
                    "max_percent": max(cpu_values),
                    "p95_percent": np.percentile(cpu_values, 95),
                },
                "memory_usage": {
                    "average_mb": statistics.mean(memory_values),
                    "max_mb": max(memory_values),
                    "p95_mb": np.percentile(memory_values, 95),
                },
            }

        # 成本分析
        costs = [s.cost for s in relevant_snapshots if s.cost is not None]
        if costs:
            report["cost_analysis"] = {
                "total_cost": sum(costs),
                "average_cost_per_request": statistics.mean(costs),
                "cost_distribution": {
                    "min": min(costs),
                    "max": max(costs),
                    "p95": np.percentile(costs, 95),
                },
            }

        # 缓存分析
        cache_hits = [s for s in relevant_snapshots if s.cache_hit is True]
        cache_misses = [s for s in relevant_snapshots if s.cache_hit is False]

        if cache_hits or cache_misses:
            total_cache_ops = len(cache_hits) + len(cache_misses)
            report["cache_analysis"] = {
                "hit_rate": len(cache_hits) / total_cache_ops if total_cache_ops > 0 else 0,
                "total_cache_operations": total_cache_ops,
                "cache_hits": len(cache_hits),
                "cache_misses": len(cache_misses),
            }

        # 基准对比
        for baseline_name in self.baselines:
            try:
                comparison = self.compare_to_baseline(baseline_name, time_range_hours * 60)
                report["baseline_comparisons"][baseline_name] = comparison
            except Exception as e:
                report["baseline_comparisons"][baseline_name] = {"error": str(e)}

        # 生成建议
        report["recommendations"] = self._generate_report_recommendations(report)

        return report

    def _calculate_peak_hour_rps(self, snapshots: List[PerformanceSnapshot]) -> float:
        """计算峰值小时RPS"""
        if not snapshots:
            return 0.0

        # 按小时分组计算RPS
        hourly_counts = defaultdict(int)

        for snapshot in snapshots:
            hour_key = snapshot.timestamp.replace(minute=0, second=0, microsecond=0)
            hourly_counts[hour_key] += 1

        return max(hourly_counts.values()) if hourly_counts else 0.0

    def _generate_report_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """根据报告数据生成建议"""
        recommendations = []

        # 成功率建议
        success_rate = report["request_statistics"]["success_rate"]
        if success_rate < 0.95:
            recommendations.append(f"成功率仅为 {success_rate:.1%}，需要改善错误处理")

        # 响应时间建议
        response_analysis = report.get("response_time_analysis", {})
        if response_analysis:
            p95_time = response_analysis.get("p95_ms", 0)
            if p95_time > 2000:
                recommendations.append(f"P95响应时间 {p95_time:.0f}ms 过高，需要性能优化")

        # 资源使用建议
        resource_analysis = report.get("resource_usage_analysis", {})
        if resource_analysis:
            cpu_p95 = resource_analysis.get("cpu_usage", {}).get("p95_percent", 0)
            if cpu_p95 > 80:
                recommendations.append(f"CPU使用率P95达到 {cpu_p95:.1f}%，考虑扩容或优化")

        # 缓存建议
        cache_analysis = report.get("cache_analysis", {})
        if cache_analysis:
            hit_rate = cache_analysis.get("hit_rate", 0)
            if hit_rate < 0.5:
                recommendations.append(f"缓存命中率仅 {hit_rate:.1%}，考虑优化缓存策略")

        # 基准对比建议
        baseline_comparisons = report.get("baseline_comparisons", {})
        for baseline_name, comparison in baseline_comparisons.items():
            if isinstance(comparison, dict) and "overall_assessment" in comparison:
                status = comparison["overall_assessment"]["status"]
                if status == "degraded":
                    recommendations.append(f"相比基准 '{baseline_name}' 性能有所退化")

        if not recommendations:
            recommendations.append("系统性能表现良好，继续保持")

        return recommendations


# 使用示例
async def example_enhanced_metrics():
    """增强指标收集器使用示例"""

    collector = EnhancedMetricsCollector(
        collection_interval=1.0,
        max_snapshots=5000,
        baseline_file=Path("enhanced_performance_baselines.json"),
    )

    # 设置告警回调
    def alert_handler(alert):
        logger.warning(
            f"🚨 性能告警: {alert['type']} = {alert['value']:.2f} (阈值: {alert['threshold']})"
        )

    def real_time_handler(metrics):
        logger.info(
            f"📊 实时指标: CPU={metrics['cpu_percent']:.1f}% 内存={metrics['memory_percent']:.1f}% RPS={metrics['throughput_rps']:.1f}"
        )

    collector.add_alert_callback(alert_handler)
    collector.add_real_time_callback(real_time_handler)

    try:
        # 启动监控
        collector.start_monitoring()

        # 模拟请求记录
        logger.info("🔄 开始模拟请求记录...")

        for i in range(100):
            # 模拟不同类型的请求
            response_time = 100 + (i % 50) * 10  # 100-590ms
            status_code = 200 if i % 10 != 0 else 500  # 10%错误率

            collector.record_request(
                endpoint="/api/v1/tests",
                method="GET",
                response_time_ms=response_time,
                status_code=status_code,
                request_size=1024,
                response_size=2048,
                provider="openai" if i % 2 == 0 else "anthropic",
                model="gpt-4" if i % 2 == 0 else "claude-3",
                tokens_used=100 + i,
                cost=0.001 * (1 + i * 0.01),
                cache_hit=i % 3 == 0,
                cache_key=f"cache_key_{i % 10}",
                custom_metrics={"test_metric": i},
            )

            await asyncio.sleep(0.1)

        # 创建性能基准
        logger.info("🎯 创建性能基准...")
        await asyncio.sleep(2)  # 等待更多数据

        # 注意：这里需要修改为异步版本
        # baseline = await collector.create_baseline(
        #     name="test_baseline_v1",
        #     test_duration_minutes=1,
        #     target_rps=50,
        #     test_conditions={"load_type": "synthetic", "environment": "test"}
        # )

        # 生成性能报告
        logger.info("📊 生成性能报告...")
        report = collector.generate_performance_report(time_range_hours=1)

        # 保存报告
        report_file = Path("enhanced_performance_report.json")
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 增强性能报告已保存: {report_file}")

        # 显示关键指标
        req_stats = report["request_statistics"]
        logger.info(
            f"📈 请求统计: 总计{req_stats['total_requests']} 成功率{req_stats['success_rate']:.1%}"
        )

        if "response_time_analysis" in report:
            rt_stats = report["response_time_analysis"]
            logger.info(
                f"⏱️ 响应时间: 平均{rt_stats['average_ms']:.1f}ms P95{rt_stats['p95_ms']:.1f}ms"
            )

        # 显示建议
        logger.info("💡 性能优化建议:")
        for rec in report["recommendations"]:
            logger.info(f"  • {rec}")

    finally:
        collector.stop_monitoring()


if __name__ == "__main__":
    asyncio.run(example_enhanced_metrics())

#!/usr/bin/env python3
"""
Complete ARC-Easy Dataset Test with Three Models
ARC-Easy完整数据集测试 - OpenAI + Anthropic + Google Gemini三模型对比
包含智能退避机制和速率限制保护
"""

import asyncio
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from test_real_api_integration import RealAPIProvider
from test_real_multi_model_comparison import RealLLMJudge

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CompleteARCEasyTest:
    """ARC-Easy完整数据集测试"""

    def __init__(self):
        self.judge = RealLLMJudge()
        self.results = []
        self.api_call_counts = {"openai": 0, "anthropic": 0, "google": 0}
        self.last_api_call = {"openai": 0, "anthropic": 0, "google": 0}

    def load_arc_easy_dataset(self) -> List[Dict[str, Any]]:
        """加载完整ARC-Easy数据集"""

        dataset_path = Path("data/processed/arc_easy.json")
        if not dataset_path.exists():
            raise FileNotFoundError(f"ARC-Easy dataset not found: {dataset_path}")

        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 随机打乱数据确保随机性
        random.shuffle(data)
        logger.info(f"加载ARC-Easy完整数据集: {len(data)}个样本")

        return data

    async def smart_backoff(self, provider: str, base_delay: float = 1.0) -> None:
        """智能退避机制 - 根据提供商和调用频率动态调整延迟"""

        current_time = time.time()
        time_since_last = current_time - self.last_api_call[provider]
        call_count = self.api_call_counts[provider]

        # 动态计算退避延迟
        if provider == "google":
            # Google Gemini付费层级但仍需谨慎
            if call_count % 10 == 0 and call_count > 0:  # 每10次调用
                delay = base_delay * 3.0  # 3秒延迟
            elif time_since_last < 0.5:
                delay = base_delay * 2.0  # 2秒延迟
            else:
                delay = base_delay  # 1秒标准延迟

        elif provider == "anthropic":
            # Anthropic相对稳定但也需要适度延迟
            if call_count % 20 == 0 and call_count > 0:  # 每20次调用
                delay = base_delay * 2.0  # 2秒延迟
            elif time_since_last < 0.3:
                delay = base_delay * 1.5  # 1.5秒延迟
            else:
                delay = base_delay * 0.5  # 0.5秒延迟

        else:  # openai
            # OpenAI最稳定，最小延迟
            if call_count % 30 == 0 and call_count > 0:  # 每30次调用
                delay = base_delay * 1.5  # 1.5秒延迟
            else:
                delay = base_delay * 0.3  # 0.3秒延迟

        # 执行延迟
        if delay > 0:
            logger.debug(f"  {provider} 智能退避: {delay:.1f}s (调用#{call_count})")
            await asyncio.sleep(delay)

        # 更新计数器
        self.api_call_counts[provider] += 1
        self.last_api_call[provider] = time.time()

    async def run_complete_arc_easy_test(self, budget_limit: float = 10.0) -> Dict[str, Any]:
        """运行完整ARC-Easy数据集测试"""

        logger.info(f"🚀 开始ARC-Easy完整数据集测试")
        logger.info(f"数据集: ARC-Easy (科学知识)")
        logger.info(f"预算限制: ${budget_limit}")
        logger.info(f"测试模型: OpenAI + Anthropic + Google三模型随机对比")

        start_time = time.time()

        # 加载完整数据集
        try:
            test_data = self.load_arc_easy_dataset()
            total_samples = len(test_data)
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            return {"error": f"Failed to load dataset: {str(e)}"}

        results = {
            "test_info": {
                "test_name": "Complete ARC-Easy Dataset Test",
                "timestamp": datetime.now().isoformat(),
                "dataset_name": "arc_easy",
                "total_samples": total_samples,
                "budget_limit": budget_limit,
                "test_type": "Three-Model Random Comparison",
                "models": [
                    "OpenAI GPT-4o-mini",
                    "Anthropic Claude-3-haiku",
                    "Google Gemini-1.5-flash",
                ],
            },
            "dataset_analysis": {
                "by_source": {"ARC-Easy": total_samples},
                "by_category": {"science": total_samples},
                "by_difficulty": {"easy": total_samples},
            },
            "model_configs": {},
            "test_results": [],
            "summary": {},
            "performance_metrics": {
                "api_call_counts": {},
                "backoff_stats": {},
                "error_recovery": {},
            },
        }

        async with RealAPIProvider() as provider:
            # 使用所有三个提供商
            available_providers = ["openai", "anthropic", "google"]
            logger.info(f"可用API提供商: {available_providers}")

            # 记录模型配置
            for provider_name in available_providers:
                config = provider.api_configs[provider_name]
                results["model_configs"][provider_name] = {
                    "provider": config.provider,
                    "model": config.model,
                    "pricing": config.pricing,
                }

            # 初始化统计
            total_cost = 0.0
            model_stats = {
                provider: {
                    "wins": 0,
                    "requests": 0,
                    "tokens": 0,
                    "cost": 0.0,
                    "avg_response_time": 0.0,
                    "response_times": [],
                    "error_count": 0,
                }
                for provider in available_providers
            }
            ties = 0
            skipped_samples = 0

            # 批量处理设置 - 减小批量大小以更好控制进度
            batch_size = 5  # 减少批量大小，更频繁的进度更新
            total_batches = (total_samples + batch_size - 1) // batch_size

            logger.info(f"开始批量处理，共{total_batches}批，每批{batch_size}个样本")
            logger.info(f"预计处理时间: {total_batches * batch_size * 15 / 3600:.1f}小时")

            # 每100个样本保存一次中间结果
            save_interval = 100
            last_save = 0

            for batch_idx in range(total_batches):
                if total_cost >= budget_limit:
                    logger.warning(f"达到预算限制 ${budget_limit:.2f}，停止测试")
                    break

                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, total_samples)
                batch_data = test_data[batch_start:batch_end]

                logger.info(
                    f"批次 {batch_idx + 1}/{total_batches} (样本 {batch_start + 1}-{batch_end})"
                )

                # 处理当前批次
                for i, sample in enumerate(batch_data):
                    global_idx = batch_start + i + 1

                    if total_cost >= budget_limit:
                        logger.warning(f"达到预算限制，停止在样本{global_idx}")
                        break

                    logger.info(
                        f"  样本 {global_idx}/{total_samples}: {sample.get('id', f'sample_{global_idx}')}"
                    )

                    # 随机选择两个不同的提供商进行对比
                    model_pair = random.sample(available_providers, 2)
                    model_a, model_b = model_pair

                    prompt = sample.get("prompt", "")
                    if not prompt:
                        logger.warning(f"    样本无效prompt，跳过")
                        skipped_samples += 1
                        continue

                    try:
                        # 模型A响应 - 带智能退避
                        await self.smart_backoff(model_a)
                        response_start = time.time()
                        response_a = await provider.generate_response(prompt, model_a)
                        response_a_time = time.time() - response_start

                        # 模型B响应 - 带智能退避
                        await self.smart_backoff(model_b)
                        response_start = time.time()
                        response_b = await provider.generate_response(prompt, model_b)
                        response_b_time = time.time() - response_start

                        if not response_a["success"] or not response_b["success"]:
                            logger.error(f"    API调用失败，跳过样本 {global_idx}")
                            if not response_a["success"]:
                                logger.error(
                                    f"      {model_a}: {response_a.get('error', 'Unknown error')}"
                                )
                                model_stats[model_a]["error_count"] += 1
                            if not response_b["success"]:
                                logger.error(
                                    f"      {model_b}: {response_b.get('error', 'Unknown error')}"
                                )
                                model_stats[model_b]["error_count"] += 1
                            skipped_samples += 1
                            continue

                        # 更新统计
                        model_stats[model_a]["requests"] += 1
                        model_stats[model_a]["tokens"] += response_a["total_tokens"]
                        model_stats[model_a]["cost"] += response_a["cost"]
                        model_stats[model_a]["response_times"].append(response_a_time)

                        model_stats[model_b]["requests"] += 1
                        model_stats[model_b]["tokens"] += response_b["total_tokens"]
                        model_stats[model_b]["cost"] += response_b["cost"]
                        model_stats[model_b]["response_times"].append(response_b_time)

                        total_cost += response_a["cost"] + response_b["cost"]

                        # 进行评判 - 使用OpenAI作为评判者
                        await self.smart_backoff("openai", 0.5)  # 评判用较短延迟
                        responses_for_judge = {"model_a": response_a, "model_b": response_b}

                        evaluation = await self.judge.evaluate_responses(
                            prompt, responses_for_judge, provider
                        )

                        if not evaluation["success"]:
                            logger.error(f"    评判失败，记录为平局")
                            evaluation = {
                                "winner": "tie",
                                "confidence": 0.5,
                                "judge_cost": 0.0,
                                "judge_tokens": 0,
                            }
                        else:
                            total_cost += evaluation["judge_cost"]

                        # 更新胜负统计
                        winner = evaluation["winner"]
                        if winner == "model_a":
                            model_stats[model_a]["wins"] += 1
                        elif winner == "model_b":
                            model_stats[model_b]["wins"] += 1
                        else:
                            ties += 1

                        # 记录结果
                        test_result = {
                            "sample_num": global_idx,
                            "sample_data": {
                                "id": sample.get("id", f"sample_{global_idx}"),
                                "source": sample.get("source", "ARC-Easy"),
                                "category": sample.get("category", "science"),
                                "difficulty": sample.get("difficulty", "easy"),
                                "prompt_length": len(prompt),
                            },
                            "model_pair": {
                                "model_a": f"{model_a} ({response_a['model']})",
                                "model_b": f"{model_b} ({response_b['model']})",
                            },
                            "responses": {
                                "model_a": {
                                    "provider": model_a,
                                    "model": response_a["model"],
                                    "tokens": response_a["total_tokens"],
                                    "cost": response_a["cost"],
                                    "response_time": response_a_time,
                                },
                                "model_b": {
                                    "provider": model_b,
                                    "model": response_b["model"],
                                    "tokens": response_b["total_tokens"],
                                    "cost": response_b["cost"],
                                    "response_time": response_b_time,
                                },
                            },
                            "evaluation": {
                                "winner": winner,
                                "confidence": evaluation["confidence"],
                                "judge_cost": evaluation["judge_cost"],
                                "judge_tokens": evaluation["judge_tokens"],
                            },
                            "timestamp": datetime.now().isoformat(),
                        }

                        results["test_results"].append(test_result)

                        logger.info(
                            f"    结果: {winner}, 成本: ${response_a['cost'] + response_b['cost'] + evaluation.get('judge_cost', 0):.6f}"
                        )

                    except Exception as e:
                        logger.error(f"    处理样本 {global_idx} 时发生错误: {str(e)}")
                        skipped_samples += 1
                        continue

                # 批次完成后显示进度和中间保存
                completed_tests = len(results["test_results"])
                progress = (completed_tests / total_samples) * 100 if total_samples > 0 else 0
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time / progress * 100 if progress > 0 else 0
                remaining_time = estimated_total - elapsed_time

                logger.info(f"批次 {batch_idx + 1} 完成")
                logger.info(f"  总进度: {completed_tests}/{total_samples} ({progress:.1f}%)")
                logger.info(
                    f"  当前成本: ${total_cost:.6f} ({total_cost/budget_limit*100:.1f}%预算)"
                )
                logger.info(f"  已用时: {elapsed_time/60:.1f}分钟")
                logger.info(f"  预计剩余: {remaining_time/60:.1f}分钟")

                # 定期保存中间结果
                if completed_tests - last_save >= save_interval:
                    interim_results = results.copy()
                    interim_results["interim_save"] = True
                    interim_results["save_point"] = completed_tests
                    self.save_interim_results(interim_results, f"interim_{completed_tests}")
                    last_save = completed_tests
                    logger.info(f"  已保存中间结果 (样本#{completed_tests})")

        end_time = time.time()
        total_time = end_time - start_time

        # 计算平均响应时间
        for provider, stats in model_stats.items():
            if stats["response_times"]:
                stats["avg_response_time"] = sum(stats["response_times"]) / len(
                    stats["response_times"]
                )
                del stats["response_times"]  # 删除原始数据节省空间

        # 计算总体统计
        completed_tests = len(results["test_results"])

        results["summary"] = {
            "total_time": total_time,
            "completed_tests": completed_tests,
            "skipped_samples": skipped_samples,
            "completion_rate": completed_tests / total_samples if total_samples > 0 else 0,
            "total_cost": total_cost,
            "budget_used_percentage": (total_cost / budget_limit * 100) if budget_limit > 0 else 0,
            "avg_cost_per_test": total_cost / completed_tests if completed_tests > 0 else 0,
            "throughput": completed_tests / total_time if total_time > 0 else 0,
            "model_stats": model_stats,
            "ties": ties,
            "judge_stats": {
                "total_evaluations": self.judge.total_evaluations,
                "total_cost": self.judge.total_cost,
            },
        }

        # 性能指标
        results["performance_metrics"] = {
            "api_call_counts": dict(self.api_call_counts),
            "total_api_calls": sum(self.api_call_counts.values()),
            "avg_processing_time": total_time / completed_tests if completed_tests > 0 else 0,
            "cost_efficiency": completed_tests / total_cost if total_cost > 0 else 0,
        }

        return results

    def save_interim_results(self, results: Dict[str, Any], suffix: str = ""):
        """保存中间结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"arc_easy_complete_test_{suffix}_{timestamp}.json"

        results_dir = Path("logs/complete_arc_easy_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"中间结果已保存到: {filepath}")
        return filepath

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """保存最终测试结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            completed = results["summary"]["completed_tests"]
            filename = f"arc_easy_complete_dataset_{completed}samples_{timestamp}.json"

        results_dir = Path("logs/complete_arc_easy_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"最终测试结果已保存到: {filepath}")
        return filepath


async def main():
    """主函数"""
    print("🚀 ARC-Easy完整数据集测试")
    print("=" * 60)

    # 检查环境变量
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    }

    print("📋 API密钥状态:")
    available_count = 0
    for provider, key in api_keys.items():
        status = "✅ 已配置" if key else "❌ 未配置"
        if key:
            available_count += 1
        print(f"  {provider}: {status}")

    if available_count < 3:
        print(f"\n⚠️  警告: 只有{available_count}个API密钥配置，将进行可用模型的对比测试")
    else:
        print(f"\n✅ 可以进行完整三模型对比测试")

    print(f"\n🔬 测试配置:")
    print(f"  数据集: ARC-Easy完整数据集 (5,197样本)")
    print(f"  模型: OpenAI + Anthropic + Google三模型随机对比")
    print(f"  预算限制: $10.00")
    print(f"  智能退避: 已启用 (避免速率限制)")
    print(f"  中间保存: 每100样本保存一次")

    # 创建测试实例
    tester = CompleteARCEasyTest()

    print(f"\n🚀 开始完整数据集测试...")

    # 运行测试
    results = await tester.run_complete_arc_easy_test(budget_limit=10.0)

    if "error" in results:
        print(f"\n❌ 测试失败: {results['error']}")
        return

    # 保存结果
    filepath = tester.save_results(results)

    # 显示结果摘要
    summary = results["summary"]

    print(f"\n📊 测试完成摘要:")
    print(f"  总耗时: {summary['total_time']:.1f}秒 ({summary['total_time']/3600:.2f}小时)")
    print(f"  完成测试: {summary['completed_tests']}")
    print(f"  跳过样本: {summary['skipped_samples']}")
    print(f"  完成率: {summary['completion_rate']:.1%}")
    print(f"  总成本: ${summary['total_cost']:.6f}")
    print(f"  预算使用: {summary['budget_used_percentage']:.1f}%")
    print(f"  平均成本/测试: ${summary['avg_cost_per_test']:.6f}")
    print(f"  吞吐量: {summary['throughput']:.3f} 测试/秒")
    print(f"  平局次数: {summary['ties']}")

    print(f"\n🏆 模型表现:")
    for model, stats in summary["model_stats"].items():
        if stats["requests"] > 0:
            win_rate = stats["wins"] / stats["requests"] * 100 if stats["requests"] > 0 else 0
            print(f"  {model.upper()}:")
            print(f"    参与测试: {stats['requests']}次")
            print(f"    获胜: {stats['wins']}次 ({win_rate:.1f}%)")
            print(f"    Token使用: {stats['tokens']:,}")
            print(f"    成本: ${stats['cost']:.6f}")
            print(f"    平均响应时间: {stats['avg_response_time']:.3f}秒")
            print(f"    错误次数: {stats['error_count']}")

    print(f"\n📈 性能指标:")
    perf = results["performance_metrics"]
    print(f"  总API调用: {perf['total_api_calls']}")
    print(f"  平均处理时间: {perf['avg_processing_time']:.1f}秒/样本")
    print(f"  成本效率: {perf['cost_efficiency']:.1f} 样本/$")

    print(f"\n✅ ARC-Easy完整数据集测试完成！")
    print(f"📁 详细结果已保存到: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())

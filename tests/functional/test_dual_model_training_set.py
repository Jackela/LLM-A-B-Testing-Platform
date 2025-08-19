#!/usr/bin/env python3
"""
Dual Model Training Set Test
双模型完整训练集测试 - 使用OpenAI和Anthropic进行大规模对比测试
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


class DualModelTrainingSetTest:
    """双模型完整训练集测试"""

    def __init__(self):
        self.judge = RealLLMJudge()
        self.results = []

    def load_complete_dataset(
        self, dataset_name: str, sample_size: int = 100
    ) -> List[Dict[str, Any]]:
        """加载完整数据集"""

        dataset_files = {
            "arc_easy": "data/processed/arc_easy.json",
            "gsm8k": "data/processed/gsm8k.json",
            "mixed": "both",  # 混合数据集
        }

        if dataset_name not in dataset_files:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Available: {list(dataset_files.keys())}"
            )

        datasets = []

        if dataset_name == "mixed":
            # 加载混合数据集 (50% ARC-Easy + 50% GSM8K)
            half_size = sample_size // 2

            # 加载ARC-Easy
            arc_path = Path(dataset_files["arc_easy"])
            if arc_path.exists():
                with open(arc_path, "r", encoding="utf-8") as f:
                    arc_data = json.load(f)
                    arc_sample = random.sample(arc_data, min(half_size, len(arc_data)))
                    datasets.extend(arc_sample)
                    logger.info(f"加载ARC-Easy样本: {len(arc_sample)}")

            # 加载GSM8K
            gsm8k_path = Path(dataset_files["gsm8k"])
            if gsm8k_path.exists():
                with open(gsm8k_path, "r", encoding="utf-8") as f:
                    gsm8k_data = json.load(f)
                    remaining_size = sample_size - len(datasets)
                    gsm8k_sample = random.sample(gsm8k_data, min(remaining_size, len(gsm8k_data)))
                    datasets.extend(gsm8k_sample)
                    logger.info(f"加载GSM8K样本: {len(gsm8k_sample)}")
        else:
            # 加载单一数据集
            dataset_path = Path(dataset_files[dataset_name])
            if not dataset_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

            with open(dataset_path, "r", encoding="utf-8") as f:
                full_data = json.load(f)
                datasets = random.sample(full_data, min(sample_size, len(full_data)))
                logger.info(f"加载{dataset_name}样本: {len(datasets)}")

        # 随机打乱数据
        random.shuffle(datasets)
        return datasets

    async def run_dual_model_test(
        self, dataset_name: str = "mixed", sample_size: int = 100, budget_limit: float = 5.0
    ) -> Dict[str, Any]:
        """运行双模型训练集测试"""

        logger.info(f"开始双模型训练集测试")
        logger.info(f"数据集: {dataset_name}")
        logger.info(f"样本数量: {sample_size}")
        logger.info(f"预算限制: ${budget_limit}")
        logger.info(f"测试模型: OpenAI vs Anthropic")

        start_time = time.time()

        # 加载数据集
        try:
            test_data = self.load_complete_dataset(dataset_name, sample_size)
        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            return {"error": f"Failed to load dataset: {str(e)}"}

        logger.info(f"成功加载{len(test_data)}个测试样本")

        results = {
            "test_info": {
                "test_name": "Dual Model Training Set Test",
                "timestamp": datetime.now().isoformat(),
                "dataset_name": dataset_name,
                "requested_sample_size": sample_size,
                "actual_sample_size": len(test_data),
                "budget_limit": budget_limit,
                "test_type": "OpenAI vs Anthropic",
            },
            "dataset_analysis": {},
            "model_configs": {},
            "test_results": [],
            "summary": {},
        }

        # 分析数据集构成
        dataset_breakdown = {}
        category_breakdown = {}
        difficulty_breakdown = {}

        for sample in test_data:
            source = sample.get("source", "unknown")
            category = sample.get("category", "unknown")
            difficulty = sample.get("difficulty", "unknown")

            dataset_breakdown[source] = dataset_breakdown.get(source, 0) + 1
            category_breakdown[category] = category_breakdown.get(category, 0) + 1
            difficulty_breakdown[difficulty] = difficulty_breakdown.get(difficulty, 0) + 1

        results["dataset_analysis"] = {
            "by_source": dataset_breakdown,
            "by_category": category_breakdown,
            "by_difficulty": difficulty_breakdown,
        }

        logger.info(f"数据集分析:")
        logger.info(f"  按来源: {dataset_breakdown}")
        logger.info(f"  按类别: {category_breakdown}")
        logger.info(f"  按难度: {difficulty_breakdown}")

        async with RealAPIProvider() as provider:
            # 只使用OpenAI和Anthropic
            available_providers = ["openai", "anthropic"]
            logger.info(f"双模型测试提供商: {available_providers}")

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
                }
                for provider in available_providers
            }
            ties = 0

            # 批量处理设置
            batch_size = 10
            total_batches = (len(test_data) + batch_size - 1) // batch_size

            logger.info(f"开始批量处理，共{total_batches}批，每批{batch_size}个样本")

            for batch_idx in range(total_batches):
                if total_cost >= budget_limit:
                    logger.warning(f"达到预算限制 ${budget_limit}，停止测试")
                    break

                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, len(test_data))
                batch_data = test_data[batch_start:batch_end]

                logger.info(
                    f"处理第{batch_idx + 1}/{total_batches}批 (样本 {batch_start + 1}-{batch_end})"
                )

                # 处理当前批次
                for i, sample in enumerate(batch_data):
                    global_idx = batch_start + i + 1

                    if total_cost >= budget_limit:
                        logger.warning(f"达到预算限制，停止在样本{global_idx}")
                        break

                    logger.info(
                        f"  处理样本 {global_idx}/{len(test_data)}: {sample.get('id', f'sample_{global_idx}')}"
                    )

                    # 固定使用OpenAI vs Anthropic
                    model_a, model_b = "openai", "anthropic"

                    prompt = sample.get("prompt", "")
                    if not prompt:
                        logger.warning(f"样本 {global_idx} 没有有效的prompt，跳过")
                        continue

                    try:
                        # 生成响应
                        response_start = time.time()
                        response_a = await provider.generate_response(prompt, model_a)
                        response_a_time = time.time() - response_start

                        await asyncio.sleep(0.5)  # 避免速率限制

                        response_start = time.time()
                        response_b = await provider.generate_response(prompt, model_b)
                        response_b_time = time.time() - response_start

                        await asyncio.sleep(0.5)  # 避免速率限制

                        if not response_a["success"] or not response_b["success"]:
                            logger.error(f"  API调用失败，跳过样本 {global_idx}")
                            if not response_a["success"]:
                                logger.error(
                                    f"    {model_a}: {response_a.get('error', 'Unknown error')}"
                                )
                            if not response_b["success"]:
                                logger.error(
                                    f"    {model_b}: {response_b.get('error', 'Unknown error')}"
                                )
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

                        # 进行评判
                        responses_for_judge = {"model_a": response_a, "model_b": response_b}

                        evaluation = await self.judge.evaluate_responses(
                            prompt, responses_for_judge, provider
                        )
                        await asyncio.sleep(0.5)  # 避免速率限制

                        if not evaluation["success"]:
                            logger.error(f"  评判失败，跳过样本 {global_idx}")
                            continue

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
                                "source": sample.get("source", "unknown"),
                                "category": sample.get("category", "unknown"),
                                "difficulty": sample.get("difficulty", "unknown"),
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
                            f"    结果: {winner}, 置信度: {evaluation['confidence']:.3f}, 成本: ${response_a['cost'] + response_b['cost'] + evaluation['judge_cost']:.6f}"
                        )

                    except Exception as e:
                        logger.error(f"  处理样本 {global_idx} 时发生错误: {str(e)}")
                        continue

                # 批次完成后显示进度
                completed_tests = len(results["test_results"])
                progress = (completed_tests / sample_size) * 100 if sample_size > 0 else 0
                logger.info(
                    f"批次 {batch_idx + 1} 完成，总进度: {completed_tests}/{sample_size} ({progress:.1f}%), 当前成本: ${total_cost:.6f}"
                )

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
            "completion_rate": completed_tests / sample_size if sample_size > 0 else 0,
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

        return results

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """保存测试结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = results["test_info"]["dataset_name"]
            sample_size = results["test_info"]["actual_sample_size"]
            filename = (
                f"dual_model_training_set_{dataset_name}_{sample_size}samples_{timestamp}.json"
            )

        # 创建结果目录
        results_dir = Path("logs/dual_model_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"测试结果已保存到: {filepath}")
        return filepath


async def main():
    """主函数"""
    print("🚀 双模型完整训练集测试")
    print("=" * 60)

    # 检查环境变量
    api_keys = {"OpenAI": os.getenv("OPENAI_API_KEY"), "Anthropic": os.getenv("ANTHROPIC_API_KEY")}

    print("📋 API密钥状态:")
    available_count = 0
    for provider, key in api_keys.items():
        status = "✅ 已配置" if key else "❌ 未配置"
        if key:
            available_count += 1
        print(f"  {provider}: {status}")

    if available_count < 2:
        print(f"\n❌ 错误: 需要OpenAI和Anthropic两个API密钥！当前只有{available_count}个")
        return

    print(f"\n✅ 可以进行双模型对比测试")

    # 测试配置选项
    test_options = {
        "1": {
            "dataset": "arc_easy",
            "size": 50,
            "budget": 3.0,
            "desc": "ARC-Easy科学数据集 (50样本)",
        },
        "2": {"dataset": "gsm8k", "size": 50, "budget": 3.0, "desc": "GSM8K数学数据集 (50样本)"},
        "3": {"dataset": "mixed", "size": 100, "budget": 5.0, "desc": "混合数据集 (100样本)"},
        "4": {
            "dataset": "mixed",
            "size": 200,
            "budget": 10.0,
            "desc": "大规模混合数据集 (200样本)",
        },
    }

    print(f"\n📊 选择测试配置:")
    for key, config in test_options.items():
        print(f"  {key}. {config['desc']} - 预算${config['budget']}")

    # 选择较小的测试配置以快速完成
    selected_config = test_options["1"]
    print(f"\n🔬 运行测试配置: {selected_config['desc']}")
    print(f"  测试模型: OpenAI GPT-4o-mini vs Anthropic Claude-3-haiku")

    # 创建测试实例
    tester = DualModelTrainingSetTest()

    print(f"\n开始测试...")
    print(f"  数据集: {selected_config['dataset']}")
    print(f"  样本数量: {selected_config['size']}")
    print(f"  预算限制: ${selected_config['budget']}")

    # 运行测试
    results = await tester.run_dual_model_test(
        dataset_name=selected_config["dataset"],
        sample_size=selected_config["size"],
        budget_limit=selected_config["budget"],
    )

    if "error" in results:
        print(f"\n❌ 测试失败: {results['error']}")
        return

    # 保存结果
    filepath = tester.save_results(results)

    # 显示结果摘要
    summary = results["summary"]
    dataset_analysis = results["dataset_analysis"]

    print(f"\n📊 测试摘要:")
    print(f"  总耗时: {summary['total_time']:.2f}秒 ({summary['total_time']/60:.1f}分钟)")
    print(f"  完成测试: {summary['completed_tests']}")
    print(f"  完成率: {summary['completion_rate']:.1%}")
    print(f"  总成本: ${summary['total_cost']:.6f}")
    print(f"  预算使用: {summary['budget_used_percentage']:.1f}%")
    print(f"  平均成本/测试: ${summary['avg_cost_per_test']:.6f}")
    print(f"  吞吐量: {summary['throughput']:.3f} 测试/秒")
    print(f"  平局次数: {summary['ties']}")

    print(f"\n📊 数据集分析:")
    print(f"  按来源: {dataset_analysis['by_source']}")
    print(f"  按类别: {dataset_analysis['by_category']}")
    print(f"  按难度: {dataset_analysis['by_difficulty']}")

    print(f"\n🏆 模型表现:")
    for model, stats in summary["model_stats"].items():
        if stats["requests"] > 0:
            win_rate = stats["wins"] / stats["requests"] * 100 if stats["requests"] > 0 else 0
            print(f"  {model.upper()}:")
            print(f"    参与测试: {stats['requests']}次")
            print(f"    获胜: {stats['wins']}次 ({win_rate:.1f}%)")
            print(f"    Token使用: {stats['tokens']}")
            print(f"    成本: ${stats['cost']:.6f}")
            print(f"    平均响应时间: {stats['avg_response_time']:.3f}秒")

    print(f"\n✅ 双模型训练集测试完成！")
    print(f"📁 详细结果已保存到: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())

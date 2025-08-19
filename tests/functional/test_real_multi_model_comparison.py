#!/usr/bin/env python3
"""
Real Multi-Model Comparison Test
真实多模型横向对比测试 - 使用真实API进行3-4个不同模型的全面对比
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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RealLLMJudge:
    """真实LLM评判系统 - 使用真实OpenAI API作为评判者"""

    def __init__(self):
        self.judge_model = "gpt-4o-mini"
        self.total_evaluations = 0
        self.total_cost = 0.0

    async def evaluate_responses(
        self, prompt: str, responses: Dict[str, Dict], provider: RealAPIProvider
    ) -> Dict[str, Any]:
        """评估两个响应的质量"""

        # 构建评估提示
        evaluation_prompt = f"""
你是一个专业的AI响应评估专家。请客观评估以下两个AI模型对同一问题的回答质量。

原始问题: {prompt}

模型A回答 ({responses['model_a']['provider']} - {responses['model_a']['model']}):
{responses['model_a']['content']}

模型B回答 ({responses['model_b']['provider']} - {responses['model_b']['model']}):
{responses['model_b']['content']}

请从以下四个维度评估 (每个维度1-10分):
1. 准确性 (Accuracy): 回答是否正确、事实准确
2. 清晰度 (Clarity): 表达是否清楚、易懂
3. 完整性 (Completeness): 回答是否完整、全面
4. 效率性 (Efficiency): 回答是否简洁、切中要点

请以JSON格式返回评估结果:
{{
  "scores": {{
    "model_a": {{
      "accuracy": [1-10分],
      "clarity": [1-10分], 
      "completeness": [1-10分],
      "efficiency": [1-10分]
    }},
    "model_b": {{
      "accuracy": [1-10分],
      "clarity": [1-10分],
      "completeness": [1-10分],
      "efficiency": [1-10分]
    }}
  }},
  "winner": "[model_a|model_b|tie]",
  "confidence": [0.0-1.0],
  "reasoning": "详细的评估理由和分析"
}}
"""

        # 调用评判API
        judge_result = await provider.generate_response(evaluation_prompt, "openai")

        if not judge_result["success"]:
            logger.error(f"Judge API调用失败: {judge_result['error']}")
            return {"success": False, "error": judge_result["error"]}

        # 解析评判结果
        try:
            # 尝试从响应中提取JSON
            content = judge_result["content"].strip()

            # 查找JSON内容
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_content = content[json_start:json_end]
                evaluation = json.loads(json_content)
            else:
                raise ValueError("No valid JSON found in response")

            # 计算加权分数
            weights = {"accuracy": 0.35, "clarity": 0.25, "completeness": 0.25, "efficiency": 0.15}

            weighted_scores = {}
            for model in ["model_a", "model_b"]:
                score = sum(
                    evaluation["scores"][model][dimension] * weight
                    for dimension, weight in weights.items()
                )
                weighted_scores[model] = round(score, 2)

            self.total_evaluations += 1
            self.total_cost += judge_result["cost"]

            return {
                "success": True,
                "winner": evaluation["winner"],
                "confidence": evaluation["confidence"],
                "reasoning": evaluation["reasoning"],
                "scores": evaluation["scores"],
                "weighted_scores": weighted_scores,
                "judge_cost": judge_result["cost"],
                "judge_tokens": judge_result["total_tokens"],
                "judge_reasoning": judge_result["content"],
            }

        except Exception as e:
            logger.error(f"解析评判结果失败: {str(e)}")
            logger.error(f"原始响应: {judge_result['content']}")

            # 返回默认评估结果
            return {
                "success": True,
                "winner": "tie",
                "confidence": 0.5,
                "reasoning": "评估系统解析失败，默认为平局",
                "scores": {
                    "model_a": {"accuracy": 5, "clarity": 5, "completeness": 5, "efficiency": 5},
                    "model_b": {"accuracy": 5, "clarity": 5, "completeness": 5, "efficiency": 5},
                },
                "weighted_scores": {"model_a": 5.0, "model_b": 5.0},
                "judge_cost": judge_result["cost"],
                "judge_tokens": judge_result["total_tokens"],
                "judge_reasoning": f"解析错误: {str(e)}",
            }


class RealMultiModelComparison:
    """真实多模型对比测试"""

    def __init__(self):
        self.judge = RealLLMJudge()
        self.results = []

    def load_test_dataset(self, sample_size: int = 20) -> List[Dict[str, str]]:
        """加载测试数据集"""

        # 从真实数据集文件加载
        datasets = []

        # 尝试加载ARC-Easy数据
        arc_path = Path("data/processed/ARC-Easy_standardized.json")
        if arc_path.exists():
            with open(arc_path, "r", encoding="utf-8") as f:
                arc_data = json.load(f)
                datasets.extend(arc_data[: sample_size // 2])

        # 尝试加载GSM8K数据
        gsm8k_path = Path("data/processed/GSM8K_standardized.json")
        if gsm8k_path.exists():
            with open(gsm8k_path, "r", encoding="utf-8") as f:
                gsm8k_data = json.load(f)
                datasets.extend(gsm8k_data[: sample_size // 2])

        # 如果没有数据集文件，使用预设的测试问题
        if not datasets:
            test_questions = [
                {
                    "id": "test_001",
                    "prompt": "What is the capital of France and what is it famous for?",
                    "category": "geography",
                    "difficulty": "easy",
                },
                {
                    "id": "test_002",
                    "prompt": "Explain the concept of photosynthesis and its importance.",
                    "category": "science",
                    "difficulty": "medium",
                },
                {
                    "id": "test_003",
                    "prompt": "Calculate 157 × 23 and show your work.",
                    "category": "math",
                    "difficulty": "easy",
                },
                {
                    "id": "test_004",
                    "prompt": "What are the main causes of climate change?",
                    "category": "science",
                    "difficulty": "medium",
                },
                {
                    "id": "test_005",
                    "prompt": "Describe the water cycle in nature.",
                    "category": "science",
                    "difficulty": "easy",
                },
                {
                    "id": "test_006",
                    "prompt": "What is the difference between renewable and non-renewable energy?",
                    "category": "science",
                    "difficulty": "medium",
                },
                {
                    "id": "test_007",
                    "prompt": "Solve for x: 2x + 5 = 17",
                    "category": "math",
                    "difficulty": "easy",
                },
                {
                    "id": "test_008",
                    "prompt": "What are the three branches of government and their functions?",
                    "category": "civics",
                    "difficulty": "medium",
                },
                {
                    "id": "test_009",
                    "prompt": "Explain the process of cellular respiration.",
                    "category": "science",
                    "difficulty": "hard",
                },
                {
                    "id": "test_010",
                    "prompt": "What is the significance of the Magna Carta?",
                    "category": "history",
                    "difficulty": "medium",
                },
            ]

            datasets = test_questions[:sample_size]

        # 随机打乱数据
        random.shuffle(datasets)
        return datasets[:sample_size]

    async def run_comparison(
        self, sample_size: int = 10, budget_limit: float = 2.0
    ) -> Dict[str, Any]:
        """运行多模型对比测试"""

        logger.info(f"开始真实多模型对比测试 - {sample_size}个样本，预算限制${budget_limit}")

        start_time = time.time()

        # 加载测试数据
        test_data = self.load_test_dataset(sample_size)
        logger.info(f"加载了{len(test_data)}个测试样本")

        results = {
            "test_info": {
                "test_name": "Real Multi-Model Comparison Test",
                "timestamp": datetime.now().isoformat(),
                "sample_size": len(test_data),
                "budget_limit": budget_limit,
            },
            "model_configs": {},
            "comparison_results": [],
            "summary": {},
        }

        async with RealAPIProvider() as provider:
            # 检查可用的API提供商
            available_providers = list(provider.api_configs.keys())
            logger.info(f"可用的API提供商: {available_providers}")

            if len(available_providers) < 2:
                logger.error("需要至少2个API提供商才能进行对比测试")
                return {
                    "error": "Insufficient API providers for comparison",
                    "available_providers": available_providers,
                    "required_minimum": 2,
                }

            # 记录模型配置
            for provider_name in available_providers:
                config = provider.api_configs[provider_name]
                results["model_configs"][provider_name] = {
                    "provider": config.provider,
                    "model": config.model,
                    "pricing": config.pricing,
                }

            # 运行对比测试
            total_cost = 0.0
            model_stats = {
                provider: {"wins": 0, "requests": 0, "tokens": 0, "cost": 0.0}
                for provider in available_providers
            }
            ties = 0

            for i, sample in enumerate(test_data):
                if total_cost >= budget_limit:
                    logger.warning(f"达到预算限制 ${budget_limit}，停止测试")
                    break

                logger.info(f"处理样本 {i+1}/{len(test_data)}: {sample.get('id', f'sample_{i+1}')}")

                # 随机选择两个不同的提供商进行对比
                model_pair = random.sample(available_providers, 2)
                model_a, model_b = model_pair

                prompt = sample.get("prompt", sample.get("question", ""))
                if not prompt:
                    logger.warning(f"样本 {i+1} 没有有效的prompt，跳过")
                    continue

                # 生成响应
                logger.info(f"  调用 {model_a} API...")
                response_a = await provider.generate_response(prompt, model_a)
                await asyncio.sleep(0.5)  # 避免速率限制

                logger.info(f"  调用 {model_b} API...")
                response_b = await provider.generate_response(prompt, model_b)
                await asyncio.sleep(0.5)  # 避免速率限制

                if not response_a["success"] or not response_b["success"]:
                    logger.error(f"  API调用失败，跳过样本 {i+1}")
                    continue

                # 更新统计
                model_stats[model_a]["requests"] += 1
                model_stats[model_a]["tokens"] += response_a["total_tokens"]
                model_stats[model_a]["cost"] += response_a["cost"]

                model_stats[model_b]["requests"] += 1
                model_stats[model_b]["tokens"] += response_b["total_tokens"]
                model_stats[model_b]["cost"] += response_b["cost"]

                total_cost += response_a["cost"] + response_b["cost"]

                # 进行评判
                logger.info(f"  进行LLM评判...")
                responses_for_judge = {"model_a": response_a, "model_b": response_b}

                evaluation = await self.judge.evaluate_responses(
                    prompt, responses_for_judge, provider
                )
                await asyncio.sleep(0.5)  # 避免速率限制

                if not evaluation["success"]:
                    logger.error(f"  评判失败，跳过样本 {i+1}")
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
                comparison_result = {
                    "sample_num": i + 1,
                    "sample_data": sample,
                    "model_pair": {
                        "model_a": f"{model_a} ({response_a['model']})",
                        "model_b": f"{model_b} ({response_b['model']})",
                    },
                    "responses": {
                        "model_a": {
                            "provider": model_a,
                            "model": response_a["model"],
                            "content": response_a["content"],
                            "tokens": response_a["total_tokens"],
                            "cost": response_a["cost"],
                        },
                        "model_b": {
                            "provider": model_b,
                            "model": response_b["model"],
                            "content": response_b["content"],
                            "tokens": response_b["total_tokens"],
                            "cost": response_b["cost"],
                        },
                    },
                    "evaluation": evaluation,
                    "timestamp": datetime.now().isoformat(),
                }

                results["comparison_results"].append(comparison_result)

                logger.info(
                    f"  结果: {winner}, 置信度: {evaluation['confidence']:.3f}, 成本: ${response_a['cost'] + response_b['cost'] + evaluation['judge_cost']:.6f}"
                )

        end_time = time.time()
        total_time = end_time - start_time

        # 计算总体统计
        completed_comparisons = len(results["comparison_results"])

        results["summary"] = {
            "total_time": total_time,
            "completed_comparisons": completed_comparisons,
            "total_cost": total_cost,
            "budget_used_percentage": (total_cost / budget_limit * 100) if budget_limit > 0 else 0,
            "avg_cost_per_comparison": (
                total_cost / completed_comparisons if completed_comparisons > 0 else 0
            ),
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
            filename = f"real_multi_model_comparison_{timestamp}.json"

        # 创建结果目录
        results_dir = Path("logs/multi_model_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"测试结果已保存到: {filepath}")
        return filepath


async def main():
    """主函数"""
    print("🚀 真实多模型横向对比测试")
    print("=" * 50)

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

    if available_count < 2:
        print(f"\n❌ 错误: 需要至少2个API密钥进行对比测试！当前只有{available_count}个")
        print("请设置以下环境变量:")
        print("  export OPENAI_API_KEY='your_openai_key'")
        print("  export ANTHROPIC_API_KEY='your_anthropic_key'")
        print("  export GOOGLE_API_KEY='your_google_key'")
        return

    print(f"\n✅ 可以进行对比测试 (可用提供商: {available_count}个)")

    # 运行测试
    comparison = RealMultiModelComparison()

    print("\n🔬 开始多模型对比测试...")
    print("  样本数量: 5个")
    print("  预算限制: $1.00")
    print("  评判模型: gpt-4o-mini")

    results = await comparison.run_comparison(sample_size=5, budget_limit=1.0)

    if "error" in results:
        print(f"\n❌ 测试失败: {results['error']}")
        return

    # 保存结果
    filepath = comparison.save_results(results)

    # 显示结果摘要
    summary = results["summary"]
    print(f"\n📊 测试摘要:")
    print(f"  总耗时: {summary['total_time']:.2f}秒")
    print(f"  完成对比: {summary['completed_comparisons']}个")
    print(f"  总成本: ${summary['total_cost']:.6f}")
    print(f"  预算使用: {summary['budget_used_percentage']:.1f}%")
    print(f"  平均成本/对比: ${summary['avg_cost_per_comparison']:.6f}")
    print(f"  平局次数: {summary['ties']}")

    print(f"\n🏆 模型表现:")
    for model, stats in summary["model_stats"].items():
        if stats["requests"] > 0:
            win_rate = stats["wins"] / stats["requests"] * 100 if stats["requests"] > 0 else 0
            print(f"  {model.upper()}:")
            print(f"    参与对比: {stats['requests']}次")
            print(f"    获胜: {stats['wins']}次 ({win_rate:.1f}%)")
            print(f"    Token使用: {stats['tokens']}")
            print(f"    成本: ${stats['cost']:.6f}")

    print(f"\n✅ 测试完成！结果已保存到: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())

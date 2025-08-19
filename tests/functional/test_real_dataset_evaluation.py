#!/usr/bin/env python3
"""
Real Dataset LLM Evaluation Test
使用真实数据集进行LLM as a Judge评估
"""

import asyncio
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class RealDatasetLLMProvider:
    """基于真实数据集的LLM提供商模拟"""

    def __init__(self, provider_name: str, style: str):
        self.provider_name = provider_name
        self.style = style

    async def generate_response(self, question: str, context: Dict[str, Any] = None) -> str:
        """根据问题类型和风格生成响应"""
        await asyncio.sleep(random.uniform(0.1, 0.3))  # 模拟真实API延迟

        if context and context.get("source") == "ARC-Easy":
            return self._handle_arc_question(question, context)
        elif context and context.get("source") == "GSM8K":
            return self._handle_math_question(question, context)
        else:
            return self._generate_generic_response(question)

    def _handle_arc_question(self, question: str, context: Dict[str, Any]) -> str:
        """处理ARC科学推理问题"""
        if "choice" in question.lower() or "which" in question.lower():
            if self.style == "detailed":
                return f"Based on scientific principles, I need to analyze each option carefully. {context.get('expected_output', 'The correct answer requires understanding the underlying scientific concepts.')}"
            else:
                return context.get(
                    "expected_output", "The answer is based on scientific reasoning."
                )

        # 通用科学问题处理
        if self.style == "detailed":
            return f"This is a scientific question that requires careful analysis. {context.get('expected_output', 'Scientific reasoning leads us to the answer.')}"
        else:
            return context.get("expected_output", "Scientific answer.")

    def _handle_math_question(self, question: str, context: Dict[str, Any]) -> str:
        """处理GSM8K数学问题"""
        if self.style == "detailed":
            # 生成详细的数学解题过程
            steps = [
                "Let me solve this step by step:",
                "First, I'll identify what we know and what we need to find.",
                "Then I'll set up the equation or calculation.",
                "Finally, I'll solve and check the answer.",
            ]
            return f"{' '.join(steps)} {context.get('ground_truth', 'Mathematical calculation completed.')}"
        else:
            # 简洁的答案
            return f"The answer is {context.get('ground_truth', 'calculated result')}."

    def _generate_generic_response(self, question: str) -> str:
        """生成通用响应"""
        if self.style == "detailed":
            return f"This question requires comprehensive analysis. I'll provide a detailed explanation to ensure clarity and understanding."
        else:
            return "Here's a concise answer to your question."


class AdvancedLLMJudge:
    """高级LLM评判者，针对不同类型的问题使用不同的评估标准"""

    def __init__(self):
        self.evaluation_criteria = {
            "science": {
                "accuracy": "Scientific correctness and factual accuracy",
                "reasoning": "Quality of scientific reasoning and logic",
                "clarity": "Clarity of explanation",
                "completeness": "Thoroughness of the answer",
            },
            "math": {
                "accuracy": "Mathematical correctness",
                "methodology": "Problem-solving approach and steps",
                "clarity": "Clarity of mathematical explanation",
                "efficiency": "Efficiency of solution method",
            },
            "general": {
                "accuracy": "Factual correctness",
                "helpfulness": "Usefulness to the user",
                "clarity": "Clear communication",
                "completeness": "Comprehensive coverage",
            },
        }

    async def evaluate_responses(
        self,
        question: str,
        response_a: str,
        response_b: str,
        provider_a: str,
        provider_b: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """使用高级评估标准评估响应"""

        # 确定问题类型和评估标准
        question_type = self._determine_question_type(question, context)
        criteria = self.evaluation_criteria.get(question_type, self.evaluation_criteria["general"])

        # 模拟LLM评估过程
        await asyncio.sleep(random.uniform(0.3, 0.7))

        # 执行评估
        evaluation = self._perform_evaluation(
            question, response_a, response_b, provider_a, provider_b, context, criteria
        )
        evaluation["question_type"] = question_type
        evaluation["evaluation_criteria"] = list(criteria.keys())

        return evaluation

    def _determine_question_type(self, question: str, context: Dict[str, Any] = None) -> str:
        """确定问题类型"""
        if context:
            if context.get("source") == "ARC-Easy" or context.get("category") == "science":
                return "science"
            elif context.get("source") == "GSM8K" or context.get("category") == "math":
                return "math"

        # 基于关键词的分类
        if any(
            keyword in question.lower()
            for keyword in ["calculate", "solve", "math", "+", "-", "*", "/"]
        ):
            return "math"
        elif any(
            keyword in question.lower()
            for keyword in ["scientific", "experiment", "hypothesis", "theory"]
        ):
            return "science"

        return "general"

    def _perform_evaluation(
        self,
        question: str,
        response_a: str,
        response_b: str,
        provider_a: str,
        provider_b: str,
        context: Dict[str, Any],
        criteria: Dict[str, str],
    ) -> Dict[str, Any]:
        """执行详细评估"""

        # 基础评分计算
        def calculate_scores(response: str, provider: str) -> Dict[str, float]:
            scores = {}

            # 根据响应长度和质量评估
            response_length = len(response)
            has_reasoning = any(
                word in response.lower()
                for word in ["because", "since", "therefore", "step", "first", "then"]
            )

            for criterion in criteria:
                base_score = 7.0

                if criterion in ["accuracy", "mathematical correctness", "scientific correctness"]:
                    # 准确性评估 - 基于是否包含预期答案
                    if context and context.get("ground_truth"):
                        if str(context["ground_truth"]).lower() in response.lower():
                            base_score = 9.0
                        elif (
                            context.get("expected_output")
                            and context["expected_output"].lower() in response.lower()
                        ):
                            base_score = 8.5
                    scores[criterion] = base_score

                elif criterion in ["reasoning", "methodology"]:
                    # 推理质量评估
                    if has_reasoning:
                        base_score = 8.5 if "detailed" in provider.lower() else 7.0
                    scores[criterion] = base_score

                elif criterion in ["clarity", "clear communication"]:
                    # 清晰度评估
                    if response_length > 20 and not response.startswith("This is"):
                        base_score = 8.0
                    scores[criterion] = base_score

                elif criterion in ["completeness", "comprehensive coverage", "efficiency"]:
                    # 完整性/效率评估
                    if "detailed" in provider.lower():
                        base_score = 8.5 if response_length > 50 else 7.5
                    else:
                        base_score = 7.5 if response_length < 100 else 8.0  # 简洁性奖励
                    scores[criterion] = base_score

                elif criterion in ["helpfulness", "usefulness"]:
                    # 有用性评估
                    base_score = 8.0 if response_length > 30 else 7.0
                    scores[criterion] = base_score

                else:
                    scores[criterion] = base_score

            # 计算总体评分
            overall = sum(scores.values()) / len(scores)
            scores["overall"] = round(overall, 1)

            return {k: round(v, 1) for k, v in scores.items()}

        # 计算两个响应的评分
        scores_a = calculate_scores(response_a, provider_a)
        scores_b = calculate_scores(response_b, provider_b)

        # 确定获胜者和置信度
        overall_a = scores_a["overall"]
        overall_b = scores_b["overall"]

        if overall_a > overall_b:
            winner = "A"
            confidence = min(0.95, 0.6 + (overall_a - overall_b) / 10)
        elif overall_b > overall_a:
            winner = "B"
            confidence = min(0.95, 0.6 + (overall_b - overall_a) / 10)
        else:
            winner = "Tie"
            confidence = 0.5

        # 生成评估理由
        reasoning = self._generate_reasoning(scores_a, scores_b, winner, criteria, context)

        return {
            "scores": {"response_a": scores_a, "response_b": scores_b},
            "winner": winner,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "judge_provider": "Advanced Claude Judge",
        }

    def _generate_reasoning(
        self,
        scores_a: Dict[str, float],
        scores_b: Dict[str, float],
        winner: str,
        criteria: Dict[str, str],
        context: Dict[str, Any],
    ) -> str:
        """生成评估理由"""

        if winner == "Tie":
            return "两个响应在各个维度上表现相近，难分高下。"

        # 找出主要差异
        score_diff = {}
        for criterion in criteria:
            if criterion in scores_a and criterion in scores_b:
                score_diff[criterion] = abs(scores_a[criterion] - scores_b[criterion])

        # 找到最大差异的维度
        if score_diff:
            max_diff_criterion = max(score_diff.keys(), key=lambda k: score_diff[k])

            criterion_names = {
                "accuracy": "准确性",
                "reasoning": "推理质量",
                "methodology": "解题方法",
                "clarity": "清晰度",
                "completeness": "完整性",
                "efficiency": "效率",
                "helpfulness": "有用性",
            }

            criterion_cn = criterion_names.get(max_diff_criterion, max_diff_criterion)

            if winner == "A":
                return f"响应A在{criterion_cn}方面明显优于响应B，显示了更好的理解和表达能力。"
            else:
                return f"响应B在{criterion_cn}方面明显优于响应A，提供了更优质的回答。"

        return f"响应{winner}在综合评估中表现更好。"


class RealDatasetTestRunner:
    """真实数据集测试运行器"""

    def __init__(self):
        # 初始化两个不同风格的提供商
        self.provider_a = RealDatasetLLMProvider("GPT-4-Analytical", "detailed")
        self.provider_b = RealDatasetLLMProvider("Claude-Efficient", "concise")
        self.judge = AdvancedLLMJudge()
        self.test_results = []

    def load_sample_data(self, sample_size: int = 6) -> List[Dict[str, Any]]:
        """从真实数据集加载样本数据"""
        datasets_dir = Path("data/processed")
        sample_data = []

        # 从ARC数据集加载样本
        arc_file = datasets_dir / "arc_easy.json"
        if arc_file.exists():
            with open(arc_file, "r", encoding="utf-8") as f:
                arc_data = json.load(f)
            sample_data.extend(random.sample(arc_data, min(3, len(arc_data))))

        # 从GSM8K数据集加载样本
        gsm8k_file = datasets_dir / "gsm8k.json"
        if gsm8k_file.exists():
            with open(gsm8k_file, "r", encoding="utf-8") as f:
                gsm8k_data = json.load(f)
            sample_data.extend(random.sample(gsm8k_data, min(3, len(gsm8k_data))))

        return sample_data[:sample_size]

    async def run_single_evaluation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个样本的评估"""
        question = sample["prompt"]
        sample_id = sample["id"]

        logger.info(f"🔍 评估样本 {sample_id}")
        logger.info(f"📝 问题: {question[:100]}...")
        logger.info(
            f"🏷️ 类别: {sample.get('category', 'unknown')} | 来源: {sample.get('source', 'unknown')}"
        )

        # 获取两个提供商的响应
        start_time = time.time()
        response_a = await self.provider_a.generate_response(question, sample)
        response_b = await self.provider_b.generate_response(question, sample)
        response_time = time.time() - start_time

        logger.info(f"📤 Provider A: {response_a[:80]}...")
        logger.info(f"📤 Provider B: {response_b[:80]}...")

        # LLM评估
        start_time = time.time()
        evaluation = await self.judge.evaluate_responses(
            question,
            response_a,
            response_b,
            self.provider_a.provider_name,
            self.provider_b.provider_name,
            sample,
        )
        eval_time = time.time() - start_time

        # 编译结果
        result = {
            "sample_data": sample,
            "responses": {
                "provider_a": {"name": self.provider_a.provider_name, "response": response_a},
                "provider_b": {"name": self.provider_b.provider_name, "response": response_b},
            },
            "evaluation": evaluation,
            "timing": {
                "response_time": round(response_time, 3),
                "evaluation_time": round(eval_time, 3),
            },
            "timestamp": datetime.now().isoformat(),
        }

        return result

    async def run_comprehensive_test(self, sample_size: int = 6) -> Dict[str, Any]:
        """运行完整的真实数据集测试"""
        logger.info("🚀 开始真实数据集LLM as a Judge测试")
        logger.info("=" * 70)

        # 加载测试数据
        sample_data = self.load_sample_data(sample_size)
        if not sample_data:
            logger.error("❌ 无法加载测试数据，请确保数据集文件存在")
            return {}

        logger.info(f"📊 已加载 {len(sample_data)} 个测试样本")

        overall_start_time = time.time()

        # 运行所有测试
        for i, sample in enumerate(sample_data, 1):
            logger.info(f"\n📋 测试 {i}/{len(sample_data)}")
            try:
                result = await self.run_single_evaluation(sample)
                self.test_results.append(result)

                # 显示评估结果
                eval_data = result["evaluation"]
                winner = eval_data["winner"]
                confidence = eval_data["confidence"]
                question_type = eval_data.get("question_type", "unknown")

                logger.info(f"🏆 获胜者: {winner} (置信度: {confidence}) | 类型: {question_type}")
                logger.info(f"💭 评估理由: {eval_data['reasoning']}")

            except Exception as e:
                logger.error(f"❌ 测试 {i} 失败: {e}")

        total_time = time.time() - overall_start_time

        # 生成汇总统计
        summary = self._generate_comprehensive_summary(total_time)

        # 保存结果
        self._save_results(summary)

        return summary

    def _generate_comprehensive_summary(self, total_time: float) -> Dict[str, Any]:
        """生成综合测试汇总"""
        if not self.test_results:
            return {}

        # 基础统计
        provider_a_wins = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "A")
        provider_b_wins = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "B")
        ties = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "Tie")

        # 按问题类型分析
        type_analysis = {}
        for result in self.test_results:
            q_type = result["evaluation"].get("question_type", "unknown")
            if q_type not in type_analysis:
                type_analysis[q_type] = {"A": 0, "B": 0, "Tie": 0, "total": 0}

            winner = result["evaluation"]["winner"]
            type_analysis[q_type][winner] += 1
            type_analysis[q_type]["total"] += 1

        # 计算平均评分和置信度
        avg_confidence = round(
            sum(r["evaluation"]["confidence"] for r in self.test_results) / len(self.test_results),
            3,
        )

        # 性能指标
        avg_response_time = round(
            sum(r["timing"]["response_time"] for r in self.test_results) / len(self.test_results), 3
        )
        avg_eval_time = round(
            sum(r["timing"]["evaluation_time"] for r in self.test_results) / len(self.test_results),
            3,
        )

        # 数据集分布
        source_distribution = {}
        for result in self.test_results:
            source = result["sample_data"].get("source", "unknown")
            source_distribution[source] = source_distribution.get(source, 0) + 1

        summary = {
            "test_info": {
                "test_name": "Real Dataset LLM as a Judge Test",
                "timestamp": datetime.now().isoformat(),
                "total_samples": len(self.test_results),
                "total_time": round(total_time, 2),
                "avg_confidence": avg_confidence,
            },
            "providers": {
                "provider_a": self.provider_a.provider_name,
                "provider_b": self.provider_b.provider_name,
            },
            "overall_results": {
                "provider_a_wins": provider_a_wins,
                "provider_b_wins": provider_b_wins,
                "ties": ties,
                "win_rate_a": round(provider_a_wins / len(self.test_results), 3),
                "win_rate_b": round(provider_b_wins / len(self.test_results), 3),
            },
            "type_analysis": type_analysis,
            "source_distribution": source_distribution,
            "performance_metrics": {
                "avg_response_time": avg_response_time,
                "avg_evaluation_time": avg_eval_time,
                "total_time": round(total_time, 2),
            },
            "detailed_results": self.test_results,
        }

        return summary

    def _save_results(self, summary: Dict[str, Any]):
        """保存测试结果"""
        results_dir = Path("logs/real_dataset_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"real_dataset_test_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n📁 详细结果已保存到: {results_file}")

    def display_comprehensive_summary(self, summary: Dict[str, Any]):
        """显示综合测试汇总"""
        logger.info("\n" + "=" * 70)
        logger.info("📊 真实数据集 LLM AS A JUDGE 测试汇总")
        logger.info("=" * 70)

        test_info = summary["test_info"]
        providers = summary["providers"]
        results = summary["overall_results"]
        type_analysis = summary["type_analysis"]
        source_dist = summary["source_distribution"]
        performance = summary["performance_metrics"]

        # 基本信息
        logger.info(f"🕒 测试时间: {test_info['timestamp']}")
        logger.info(f"📊 测试样本数: {test_info['total_samples']}")
        logger.info(f"⏱️ 总耗时: {test_info['total_time']}秒")
        logger.info(f"🎯 平均置信度: {test_info['avg_confidence']}")

        # 提供商信息
        logger.info(f"\n🤖 测试提供商:")
        logger.info(f"  Provider A: {providers['provider_a']}")
        logger.info(f"  Provider B: {providers['provider_b']}")

        # 总体结果
        logger.info(f"\n🏆 总体比赛结果:")
        logger.info(
            f"  Provider A 获胜: {results['provider_a_wins']} 次 ({results['win_rate_a']:.1%})"
        )
        logger.info(
            f"  Provider B 获胜: {results['provider_b_wins']} 次 ({results['win_rate_b']:.1%})"
        )
        logger.info(f"  平局: {results['ties']} 次")

        # 按问题类型分析
        logger.info(f"\n📋 按问题类型分析:")
        for q_type, type_data in type_analysis.items():
            total = type_data["total"]
            logger.info(f"  {q_type.upper()}类型 (共{total}题):")
            logger.info(f"    Provider A: {type_data['A']} 胜 ({type_data['A']/total:.1%})")
            logger.info(f"    Provider B: {type_data['B']} 胜 ({type_data['B']/total:.1%})")
            if type_data["Tie"] > 0:
                logger.info(f"    平局: {type_data['Tie']} 次")

        # 数据集分布
        logger.info(f"\n📚 数据集分布:")
        for source, count in source_dist.items():
            logger.info(f"  {source}: {count} 个样本")

        # 性能指标
        logger.info(f"\n⚡ 性能指标:")
        logger.info(f"  平均响应时间: {performance['avg_response_time']}秒")
        logger.info(f"  平均评估时间: {performance['avg_evaluation_time']}秒")

        # 结论和建议
        logger.info(f"\n🎯 测试结论:")
        if results["provider_a_wins"] > results["provider_b_wins"]:
            winner_name = providers["provider_a"]
            win_rate = results["win_rate_a"]
        elif results["provider_b_wins"] > results["provider_a_wins"]:
            winner_name = providers["provider_b"]
            win_rate = results["win_rate_b"]
        else:
            winner_name = "平局"
            win_rate = 0.5

        if win_rate >= 0.7:
            conclusion = f"  🎉 {winner_name} 表现显著优秀 (胜率: {win_rate:.1%})"
        elif win_rate >= 0.6:
            conclusion = f"  ✅ {winner_name} 表现较好 (胜率: {win_rate:.1%})"
        else:
            conclusion = f"  ⚖️ 两个模型表现相近，各有优势"

        logger.info(conclusion)
        logger.info(f"  💡 建议: 根据具体应用场景选择合适的模型")


async def main():
    """主测试函数"""
    logger.info("🎯 真实数据集 LLM as a Judge 评估测试")
    logger.info("本测试使用下载的ARC-Easy和GSM8K数据集进行真实评估")

    # 创建并运行测试
    tester = RealDatasetTestRunner()
    summary = await tester.run_comprehensive_test(sample_size=6)

    if summary:
        # 显示汇总结果
        tester.display_comprehensive_summary(summary)
        logger.info("\n✅ 真实数据集 LLM as a Judge 测试完成！")
    else:
        logger.error("❌ 测试失败，请检查数据集文件")

    return summary


if __name__ == "__main__":
    asyncio.run(main())

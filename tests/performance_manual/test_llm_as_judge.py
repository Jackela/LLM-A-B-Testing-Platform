#!/usr/bin/env python3
"""
LLM as a Judge Test - Small Sample Evaluation
运行小样本测试并使用LLM评估结果
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MockLLMProvider:
    """模拟LLM提供商，用于生成测试响应"""

    def __init__(self, provider_name: str, style: str):
        self.provider_name = provider_name
        self.style = style

    async def generate_response(self, prompt: str) -> str:
        """根据不同风格生成模拟响应"""
        await asyncio.sleep(0.1)  # 模拟API调用延迟

        if "capital of France" in prompt.lower():
            if self.style == "detailed":
                return "The capital of France is Paris. Paris is located in the north-central part of France and has been the country's capital since 1789. It is known for landmarks like the Eiffel Tower and the Louvre Museum."
            else:
                return "Paris is the capital of France."

        elif "solve" in prompt.lower() and "math" in prompt.lower():
            if "2 + 3" in prompt:
                if self.style == "detailed":
                    return "To solve 2 + 3, I add the two numbers together: 2 + 3 = 5. The answer is 5."
                else:
                    return "2 + 3 = 5"
            elif "10 * 6" in prompt:
                if self.style == "detailed":
                    return "To calculate 10 × 6, I multiply: 10 × 6 = 60. The final answer is 60."
                else:
                    return "10 * 6 = 60"

        elif "explain" in prompt.lower() and "photosynthesis" in prompt.lower():
            if self.style == "detailed":
                return "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen. This process occurs in chloroplasts and is essential for plant survival and oxygen production on Earth."
            else:
                return "Photosynthesis is how plants make food using sunlight, water, and CO2."

        # Default responses
        if self.style == "detailed":
            return f"This is a detailed response from {self.provider_name}. I aim to provide comprehensive and thorough answers to help users understand the topic fully."
        else:
            return f"This is a concise response from {self.provider_name}."


class LLMJudge:
    """LLM作为评判者，评估两个响应的质量"""

    def __init__(self):
        self.judge_prompt_template = """
作为一个专业的AI评估专家，请评估以下两个AI响应的质量。

问题：{question}

响应A ({provider_a}):
{response_a}

响应B ({provider_b}):
{response_b}

请从以下维度评估（每个维度1-10分）：
1. 准确性 (Accuracy) - 信息是否正确
2. 完整性 (Completeness) - 是否充分回答了问题  
3. 清晰度 (Clarity) - 表达是否清晰易懂
4. 有用性 (Helpfulness) - 对用户是否有帮助

请提供：
1. 各维度的评分
2. 总体评分 (1-10)
3. 简要说明理由
4. 推荐哪个响应更好

请以JSON格式返回结果：
{{
    "scores": {{
        "response_a": {{
            "accuracy": 分数,
            "completeness": 分数,
            "clarity": 分数,
            "helpfulness": 分数,
            "overall": 分数
        }},
        "response_b": {{
            "accuracy": 分数,
            "completeness": 分数,
            "clarity": 分数,
            "helpfulness": 分数,
            "overall": 分数
        }}
    }},
    "winner": "A" 或 "B" 或 "Tie",
    "confidence": 置信度(0-1),
    "reasoning": "评估理由",
    "judge_provider": "Claude"
}}
"""

    async def evaluate_responses(
        self, question: str, response_a: str, response_b: str, provider_a: str, provider_b: str
    ) -> Dict[str, Any]:
        """使用LLM评估两个响应"""

        # 构建评估提示
        judge_prompt = self.judge_prompt_template.format(
            question=question,
            response_a=response_a,
            response_b=response_b,
            provider_a=provider_a,
            provider_b=provider_b,
        )

        # 模拟LLM评估（实际应用中这里会调用真实的LLM API）
        await asyncio.sleep(0.5)  # 模拟API调用时间

        # 基于启发式规则的模拟评估
        eval_result = self._simulate_evaluation(
            question, response_a, response_b, provider_a, provider_b
        )

        return eval_result

    def _simulate_evaluation(
        self, question: str, response_a: str, response_b: str, provider_a: str, provider_b: str
    ) -> Dict[str, Any]:
        """模拟LLM评估逻辑"""

        # 计算响应长度和信息密度
        len_a, len_b = len(response_a), len(response_b)

        # 基本评分逻辑
        def score_response(response: str, is_detailed: bool) -> Dict[str, float]:
            base_score = 7.0

            # 准确性评估
            accuracy = (
                8.0
                if any(
                    keyword in response.lower()
                    for keyword in ["paris", "5", "60", "photosynthesis"]
                )
                else 7.0
            )

            # 完整性评估
            completeness = 8.5 if len(response) > 50 else 6.5

            # 清晰度评估
            clarity = 8.0 if not response.lower().startswith("this is") else 7.0

            # 有用性评估
            helpfulness = 8.5 if is_detailed else 7.5

            overall = (accuracy + completeness + clarity + helpfulness) / 4

            return {
                "accuracy": round(accuracy, 1),
                "completeness": round(completeness, 1),
                "clarity": round(clarity, 1),
                "helpfulness": round(helpfulness, 1),
                "overall": round(overall, 1),
            }

        # 评估两个响应
        scores_a = score_response(response_a, "detailed" in provider_a.lower())
        scores_b = score_response(response_b, "detailed" in provider_b.lower())

        # 确定获胜者
        if scores_a["overall"] > scores_b["overall"]:
            winner = "A"
            confidence = min(0.9, (scores_a["overall"] - scores_b["overall"]) / 2 + 0.6)
        elif scores_b["overall"] > scores_a["overall"]:
            winner = "B"
            confidence = min(0.9, (scores_b["overall"] - scores_a["overall"]) / 2 + 0.6)
        else:
            winner = "Tie"
            confidence = 0.5

        # 生成评估理由
        if winner == "A":
            reasoning = f"响应A在多个维度上表现更好，特别是在{'完整性' if scores_a['completeness'] > scores_b['completeness'] else '有用性'}方面。"
        elif winner == "B":
            reasoning = f"响应B在多个维度上表现更好，特别是在{'完整性' if scores_b['completeness'] > scores_a['completeness'] else '有用性'}方面。"
        else:
            reasoning = "两个响应质量相近，各有优劣。"

        return {
            "scores": {"response_a": scores_a, "response_b": scores_b},
            "winner": winner,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "judge_provider": "Claude (Simulated)",
        }


class SmallSampleTest:
    """小样本测试运行器"""

    def __init__(self):
        self.provider_a = MockLLMProvider("GPT-4-Detailed", "detailed")
        self.provider_b = MockLLMProvider("Claude-Concise", "concise")
        self.judge = LLMJudge()
        self.test_results = []

    def get_test_questions(self) -> List[Dict[str, Any]]:
        """获取测试问题集"""
        return [
            {
                "id": "q1",
                "question": "What is the capital of France?",
                "category": "factual",
                "expected_answer": "Paris",
            },
            {
                "id": "q2",
                "question": "Solve this math problem: 2 + 3 = ?",
                "category": "math",
                "expected_answer": "5",
            },
            {
                "id": "q3",
                "question": "Calculate: 10 * 6",
                "category": "math",
                "expected_answer": "60",
            },
            {
                "id": "q4",
                "question": "Explain photosynthesis in simple terms",
                "category": "science",
                "expected_answer": "Process by which plants make food using sunlight",
            },
        ]

    async def run_evaluation(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个问题的评估"""
        question = question_data["question"]

        logger.info(f"🔍 评估问题: {question}")

        # 获取两个提供商的响应
        start_time = time.time()
        response_a = await self.provider_a.generate_response(question)
        response_b = await self.provider_b.generate_response(question)
        response_time = time.time() - start_time

        logger.info(f"📝 Provider A ({self.provider_a.provider_name}): {response_a}")
        logger.info(f"📝 Provider B ({self.provider_b.provider_name}): {response_b}")

        # LLM评估
        start_time = time.time()
        evaluation = await self.judge.evaluate_responses(
            question,
            response_a,
            response_b,
            self.provider_a.provider_name,
            self.provider_b.provider_name,
        )
        eval_time = time.time() - start_time

        # 编译结果
        result = {
            "question_data": question_data,
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

    async def run_small_sample_test(self) -> Dict[str, Any]:
        """运行完整的小样本测试"""
        logger.info("🚀 开始LLM as a Judge小样本测试")
        logger.info("=" * 60)

        test_questions = self.get_test_questions()
        overall_start_time = time.time()

        # 运行所有测试
        for i, question_data in enumerate(test_questions, 1):
            logger.info(f"\n📋 测试 {i}/{len(test_questions)}")
            result = await self.run_evaluation(question_data)
            self.test_results.append(result)

            # 显示评估结果
            eval_data = result["evaluation"]
            winner = eval_data["winner"]
            confidence = eval_data["confidence"]

            logger.info(f"🏆 获胜者: {winner} (置信度: {confidence})")
            logger.info(f"💭 评估理由: {eval_data['reasoning']}")

        total_time = time.time() - overall_start_time

        # 生成汇总统计
        summary = self._generate_summary(total_time)

        # 保存结果
        self._save_results(summary)

        return summary

    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """生成测试汇总"""
        if not self.test_results:
            return {}

        # 计算统计数据
        provider_a_wins = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "A")
        provider_b_wins = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "B")
        ties = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "Tie")

        # 平均评分
        avg_scores_a = {}
        avg_scores_b = {}
        for dimension in ["accuracy", "completeness", "clarity", "helpfulness", "overall"]:
            avg_scores_a[dimension] = round(
                sum(r["evaluation"]["scores"]["response_a"][dimension] for r in self.test_results)
                / len(self.test_results),
                2,
            )
            avg_scores_b[dimension] = round(
                sum(r["evaluation"]["scores"]["response_b"][dimension] for r in self.test_results)
                / len(self.test_results),
                2,
            )

        # 平均响应时间
        avg_response_time = round(
            sum(r["timing"]["response_time"] for r in self.test_results) / len(self.test_results), 3
        )
        avg_eval_time = round(
            sum(r["timing"]["evaluation_time"] for r in self.test_results) / len(self.test_results),
            3,
        )

        summary = {
            "test_info": {
                "test_name": "LLM as a Judge - Small Sample Test",
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(self.test_results),
                "total_time": round(total_time, 2),
            },
            "providers": {
                "provider_a": self.provider_a.provider_name,
                "provider_b": self.provider_b.provider_name,
            },
            "results": {
                "provider_a_wins": provider_a_wins,
                "provider_b_wins": provider_b_wins,
                "ties": ties,
                "win_rate_a": round(provider_a_wins / len(self.test_results), 2),
                "win_rate_b": round(provider_b_wins / len(self.test_results), 2),
            },
            "average_scores": {"provider_a": avg_scores_a, "provider_b": avg_scores_b},
            "performance": {
                "avg_response_time": avg_response_time,
                "avg_evaluation_time": avg_eval_time,
                "total_time": round(total_time, 2),
            },
            "detailed_results": self.test_results,
        }

        return summary

    def _save_results(self, summary: Dict[str, Any]):
        """保存测试结果"""
        # 创建结果目录
        results_dir = Path("logs/llm_judge_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"llm_judge_test_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n📁 详细结果已保存到: {results_file}")

    def display_summary(self, summary: Dict[str, Any]):
        """显示测试汇总"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 LLM AS A JUDGE 测试汇总")
        logger.info("=" * 60)

        test_info = summary["test_info"]
        providers = summary["providers"]
        results = summary["results"]
        avg_scores = summary["average_scores"]
        performance = summary["performance"]

        logger.info(f"🕒 测试时间: {test_info['timestamp']}")
        logger.info(f"📝 测试问题数: {test_info['total_questions']}")
        logger.info(f"⏱️ 总耗时: {test_info['total_time']}秒")

        logger.info(f"\n🤖 测试提供商:")
        logger.info(f"  Provider A: {providers['provider_a']}")
        logger.info(f"  Provider B: {providers['provider_b']}")

        logger.info(f"\n🏆 比赛结果:")
        logger.info(
            f"  Provider A 获胜: {results['provider_a_wins']} 次 ({results['win_rate_a']:.0%})"
        )
        logger.info(
            f"  Provider B 获胜: {results['provider_b_wins']} 次 ({results['win_rate_b']:.0%})"
        )
        logger.info(f"  平局: {results['ties']} 次")

        logger.info(f"\n📈 平均评分对比:")
        logger.info(f"{'维度':<12} {'Provider A':<12} {'Provider B':<12} {'差值'}")
        logger.info("-" * 50)

        for dimension in ["accuracy", "completeness", "clarity", "helpfulness", "overall"]:
            score_a = avg_scores["provider_a"][dimension]
            score_b = avg_scores["provider_b"][dimension]
            diff = round(score_a - score_b, 2)

            dim_cn = {
                "accuracy": "准确性",
                "completeness": "完整性",
                "clarity": "清晰度",
                "helpfulness": "有用性",
                "overall": "总体评分",
            }

            logger.info(f"{dim_cn[dimension]:<10} {score_a:<12} {score_b:<12} {diff:+.2f}")

        logger.info(f"\n⚡ 性能指标:")
        logger.info(f"  平均响应时间: {performance['avg_response_time']}秒")
        logger.info(f"  平均评估时间: {performance['avg_evaluation_time']}秒")

        # 结论
        if results["provider_a_wins"] > results["provider_b_wins"]:
            winner = providers["provider_a"]
        elif results["provider_b_wins"] > results["provider_a_wins"]:
            winner = providers["provider_b"]
        else:
            winner = "平局"

        logger.info(f"\n🎯 测试结论:")
        logger.info(f"  在本次小样本测试中，{winner} 表现更优秀")
        logger.info(f"  两个模型在不同维度各有优势，建议根据具体需求选择")


async def main():
    """主测试函数"""
    logger.info("🎯 LLM as a Judge - 小样本评估测试")
    logger.info("本测试将模拟两个LLM提供商的响应，并使用LLM作为评判者进行评估")

    # 创建并运行测试
    tester = SmallSampleTest()
    summary = await tester.run_small_sample_test()

    # 显示汇总结果
    tester.display_summary(summary)

    logger.info("\n✅ LLM as a Judge 测试完成！")
    return summary


if __name__ == "__main__":
    asyncio.run(main())

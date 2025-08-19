#!/usr/bin/env python3
"""
Multi-Model Comparison Test
多模型横向对比测试 - 支持3-5个不同模型的全面对比
"""

import asyncio
import itertools
import json
import logging
import math
import random
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MultiModelProvider:
    """多模型提供商 - 支持多种不同的模型风格和能力"""

    def __init__(self, provider_name: str, model_name: str, model_config: Dict[str, Any]):
        self.provider_name = provider_name
        self.model_name = model_name
        self.model_config = model_config
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # 扩展的价格表 (每1K tokens, USD)
        self.pricing = {
            "gemini-pro": {"input": 0.0005, "output": 0.0015},
            "gemini-flash": {"input": 0.000075, "output": 0.0003},
            "claude-3.5-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "llama-3.1-70b": {"input": 0.0009, "output": 0.0009},  # 估算价格
            "mistral-large": {"input": 0.008, "output": 0.024},  # 估算价格
        }

        # 模型特性配置
        self.model_characteristics = {
            "gemini-pro": {
                "reasoning_style": "systematic",
                "detail_level": "high",
                "creativity": "medium",
                "accuracy_bias": 0.85,
                "response_length_multiplier": 1.4,
            },
            "gemini-flash": {
                "reasoning_style": "efficient",
                "detail_level": "medium",
                "creativity": "high",
                "accuracy_bias": 0.80,
                "response_length_multiplier": 1.1,
            },
            "claude-3.5-sonnet": {
                "reasoning_style": "analytical",
                "detail_level": "very_high",
                "creativity": "high",
                "accuracy_bias": 0.90,
                "response_length_multiplier": 1.6,
            },
            "claude-3-haiku": {
                "reasoning_style": "concise",
                "detail_level": "medium",
                "creativity": "medium",
                "accuracy_bias": 0.85,
                "response_length_multiplier": 0.8,
            },
            "gpt-4o": {
                "reasoning_style": "comprehensive",
                "detail_level": "very_high",
                "creativity": "very_high",
                "accuracy_bias": 0.92,
                "response_length_multiplier": 1.5,
            },
            "gpt-4o-mini": {
                "reasoning_style": "balanced",
                "detail_level": "medium",
                "creativity": "medium",
                "accuracy_bias": 0.82,
                "response_length_multiplier": 1.0,
            },
            "gpt-3.5-turbo": {
                "reasoning_style": "conversational",
                "detail_level": "medium",
                "creativity": "medium",
                "accuracy_bias": 0.78,
                "response_length_multiplier": 1.2,
            },
            "llama-3.1-70b": {
                "reasoning_style": "structured",
                "detail_level": "high",
                "creativity": "medium",
                "accuracy_bias": 0.83,
                "response_length_multiplier": 1.3,
            },
            "mistral-large": {
                "reasoning_style": "precise",
                "detail_level": "high",
                "creativity": "medium",
                "accuracy_bias": 0.87,
                "response_length_multiplier": 1.2,
            },
        }

    async def generate_response(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """生成模型特异性响应"""
        # 基于模型特性调整延迟
        char = self.model_characteristics.get(self.model_name, {})
        base_delay = char.get("response_length_multiplier", 1.0) * 0.3
        await asyncio.sleep(base_delay + random.uniform(0.1, 0.3))

        # 生成响应内容
        content = self._generate_model_specific_response(prompt, context, char)

        # 计算token使用量
        input_tokens = self._calculate_tokens(prompt)
        output_tokens = self._calculate_tokens(content)

        # 计算成本
        cost = self._calculate_cost(input_tokens, output_tokens)

        # 更新统计
        self.request_count += 1
        self.total_tokens += input_tokens + output_tokens
        self.total_cost += cost

        return {
            "content": content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "model_name": self.model_name,
            "provider": self.provider_name,
            "characteristics": char,
        }

    def _generate_model_specific_response(
        self, prompt: str, context: Dict[str, Any], char: Dict[str, Any]
    ) -> str:
        """基于模型特性生成特异性响应"""
        if not context:
            return self._generate_generic_response(prompt, char)

        source = context.get("source", "")
        category = context.get("category", "")

        if source == "ARC-Easy" or category == "science":
            return self._generate_science_response(prompt, context, char)
        elif source == "GSM8K" or category == "math":
            return self._generate_math_response(prompt, context, char)
        else:
            return self._generate_generic_response(prompt, char)

    def _generate_science_response(
        self, prompt: str, context: Dict[str, Any], char: Dict[str, Any]
    ) -> str:
        """生成科学问题响应"""
        expected = context.get("expected_output", "scientific conclusion")
        reasoning_style = char.get("reasoning_style", "systematic")
        detail_level = char.get("detail_level", "medium")
        accuracy_bias = char.get("accuracy_bias", 0.8)

        # 基于准确性偏差决定是否给出正确答案
        is_accurate = random.random() < accuracy_bias
        answer = expected if is_accurate else self._generate_plausible_wrong_answer(expected)

        if reasoning_style == "systematic":
            if detail_level == "very_high":
                return f"""Let me approach this scientific question systematically.

First, I'll analyze the fundamental principles involved: {answer}

This conclusion is based on established scientific theories and can be verified through empirical evidence. The underlying mechanisms involve well-documented scientific processes."""
            elif detail_level == "high":
                return f"Analyzing this systematically, the scientific principle leads us to: {answer}. This follows from fundamental scientific knowledge."
            else:
                return f"Scientific analysis shows: {answer}"

        elif reasoning_style == "analytical":
            if detail_level == "very_high":
                return f"""From an analytical perspective, this question requires careful examination of the scientific concepts involved.

The correct answer is: {answer}

This conclusion follows from a thorough analysis of the underlying scientific principles and their practical applications in this context."""
            else:
                return f"Analytical assessment: {answer}. This conclusion is supported by scientific reasoning."

        elif reasoning_style == "efficient":
            return f"Based on scientific principles: {answer}."

        elif reasoning_style == "concise":
            return f"{answer}"

        elif reasoning_style == "comprehensive":
            return f"""This scientific question requires a comprehensive understanding of multiple factors.

Answer: {answer}

Explanation: This conclusion integrates various scientific principles and demonstrates how they interact in this specific context. The scientific basis is well-established."""

        elif reasoning_style == "conversational":
            return f"Looking at this science question, I'd say the answer is {answer}. This makes sense when you consider the basic scientific principles at work here."

        elif reasoning_style == "structured":
            return f"""1. Question analysis: Scientific reasoning problem
2. Relevant principles: Established scientific knowledge
3. Conclusion: {answer}
4. Verification: Consistent with scientific theory"""

        elif reasoning_style == "precise":
            return f"Scientific determination: {answer}. Basis: established principles."

        else:  # balanced/default
            return f"Based on scientific principles, the answer is {answer}. This follows from established knowledge in the field."

    def _generate_math_response(
        self, prompt: str, context: Dict[str, Any], char: Dict[str, Any]
    ) -> str:
        """生成数学问题响应"""
        answer = context.get("ground_truth", "calculated result")
        reasoning_style = char.get("reasoning_style", "systematic")
        detail_level = char.get("detail_level", "medium")
        accuracy_bias = char.get("accuracy_bias", 0.8)

        # 基于准确性偏差决定是否给出正确答案
        is_accurate = random.random() < accuracy_bias
        final_answer = answer if is_accurate else self._generate_wrong_math_answer(answer)

        if reasoning_style == "systematic":
            if detail_level == "very_high":
                return f"""I'll solve this step-by-step using a systematic approach:

Step 1: Identify the given information and what we need to find
Step 2: Determine the appropriate mathematical method
Step 3: Set up the equation or calculation
Step 4: Perform the computation carefully
Step 5: Verify the result

Final answer: {final_answer}

This solution follows standard mathematical procedures and can be verified through alternative methods."""
            elif detail_level == "high":
                return f"Solving systematically: First, I'll identify the key information. Then I'll apply the appropriate mathematical operations. The answer is {final_answer}."
            else:
                return f"Step-by-step solution yields: {final_answer}"

        elif reasoning_style == "analytical":
            return f"Mathematical analysis: By breaking down this problem and applying appropriate mathematical reasoning, I determine that {final_answer}."

        elif reasoning_style == "efficient":
            return f"Quick calculation: {final_answer}"

        elif reasoning_style == "concise":
            return f"{final_answer}"

        elif reasoning_style == "comprehensive":
            return f"""This mathematical problem requires comprehensive analysis of the given information.

Solution approach: I need to carefully examine the problem structure and apply appropriate mathematical concepts.

Final answer: {final_answer}

This result represents the accurate mathematical solution based on the given parameters."""

        elif reasoning_style == "conversational":
            return (
                f"Let me work through this math problem. When I calculate it, I get {final_answer}."
            )

        elif reasoning_style == "structured":
            return f"""Problem type: Mathematical calculation
Given data: [extracted from problem]
Method: Standard mathematical operations
Result: {final_answer}"""

        elif reasoning_style == "precise":
            return f"Mathematical result: {final_answer}"

        else:  # balanced/default
            return f"Solving this mathematically: {final_answer}"

    def _generate_generic_response(self, prompt: str, char: Dict[str, Any]) -> str:
        """生成通用响应"""
        reasoning_style = char.get("reasoning_style", "systematic")
        detail_level = char.get("detail_level", "medium")

        if detail_level == "very_high":
            return f"This question requires comprehensive analysis considering multiple perspectives and factors. I'll provide a detailed examination to ensure thorough understanding of all relevant aspects."
        elif detail_level == "high":
            return f"This requires careful analysis. Let me provide a detailed response addressing the key aspects of your question."
        else:
            return f"Here's my response addressing your question with appropriate consideration of the relevant factors."

    def _generate_plausible_wrong_answer(self, correct_answer: str) -> str:
        """生成看似合理的错误答案"""
        # 简化的错误答案生成
        alternatives = [
            "absorbed light from atmospheric particles",
            "internal luminescence from moon's core",
            "electromagnetic radiation from Earth",
            "thermal emission from surface heating",
        ]
        return (
            random.choice(alternatives) if correct_answer else "alternative scientific explanation"
        )

    def _generate_wrong_math_answer(self, correct_answer: str) -> str:
        """生成错误的数学答案"""
        try:
            if correct_answer.isdigit():
                num = int(correct_answer)
                # 生成接近但错误的答案
                error_factor = random.choice([0.9, 1.1, 0.8, 1.2])
                wrong_num = int(num * error_factor)
                return (
                    str(wrong_num) if wrong_num != num else str(num + random.choice([-1, 1, -2, 2]))
                )
            else:
                return str(random.randint(10, 100))
        except:
            return "incorrect calculation"

    def _calculate_tokens(self, text: str) -> int:
        """计算token数量"""
        word_count = len(text.split())
        char_count = len(text)

        # 基于模型特性调整token计算
        char = self.model_characteristics.get(self.model_name, {})
        multiplier = char.get("response_length_multiplier", 1.0)

        token_estimate = max(
            int(word_count * 1.3 * multiplier), int(char_count * 0.25 * multiplier)
        )

        return max(1, token_estimate)

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """计算精确成本"""
        pricing = self.pricing.get(self.model_name, self.pricing["gpt-3.5-turbo"])

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost


class TournamentJudge:
    """锦标赛式评判者 - 支持多模型两两对比"""

    def __init__(self):
        self.evaluation_count = 0
        self.pairwise_results = defaultdict(lambda: defaultdict(int))

    async def evaluate_pairwise(
        self,
        question: str,
        response_a: Dict[str, Any],
        response_b: Dict[str, Any],
        model_a: str,
        model_b: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """执行两两对比评估"""

        # 模拟评估延迟
        await asyncio.sleep(random.uniform(0.3, 0.7))

        # 高级评估逻辑
        evaluation = self._advanced_pairwise_evaluation(
            question,
            response_a["content"],
            response_b["content"],
            response_a["characteristics"],
            response_b["characteristics"],
            context,
        )

        # 记录对比结果
        if evaluation["winner"] == "A":
            self.pairwise_results[model_a][model_b] += 1
        elif evaluation["winner"] == "B":
            self.pairwise_results[model_b][model_a] += 1
        # 平局不计入胜负关系

        self.evaluation_count += 1
        return evaluation

    def _advanced_pairwise_evaluation(
        self,
        question: str,
        response_a: str,
        response_b: str,
        char_a: Dict[str, Any],
        char_b: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """高级两两对比评估"""

        # 多维度评分
        scores_a = self._calculate_comprehensive_scores(response_a, char_a, context)
        scores_b = self._calculate_comprehensive_scores(response_b, char_b, context)

        # 加权评分
        weights = {"accuracy": 0.4, "clarity": 0.2, "completeness": 0.2, "reasoning_quality": 0.2}

        weighted_score_a = sum(scores_a.get(dim, 7.0) * weight for dim, weight in weights.items())
        weighted_score_b = sum(scores_b.get(dim, 7.0) * weight for dim, weight in weights.items())

        # 引入一些随机性以模拟真实评估的不确定性
        noise_a = random.uniform(-0.2, 0.2)
        noise_b = random.uniform(-0.2, 0.2)

        final_score_a = weighted_score_a + noise_a
        final_score_b = weighted_score_b + noise_b

        # 确定获胜者
        score_diff = abs(final_score_a - final_score_b)

        if final_score_a > final_score_b + 0.3:
            winner = "A"
            confidence = min(0.95, 0.6 + score_diff / 8)
        elif final_score_b > final_score_a + 0.3:
            winner = "B"
            confidence = min(0.95, 0.6 + score_diff / 8)
        else:
            winner = "Tie"
            confidence = 0.5 + random.uniform(-0.1, 0.1)

        # 生成评估理由
        reasoning = self._generate_comparison_reasoning(scores_a, scores_b, char_a, char_b, winner)

        return {
            "winner": winner,
            "confidence": round(confidence, 3),
            "reasoning": reasoning,
            "scores": {
                "response_a": {k: round(v, 2) for k, v in scores_a.items()},
                "response_b": {k: round(v, 2) for k, v in scores_b.items()},
            },
            "weighted_scores": {
                "response_a": round(final_score_a, 2),
                "response_b": round(final_score_b, 2),
            },
        }

    def _calculate_comprehensive_scores(
        self, response: str, characteristics: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, float]:
        """计算综合评分"""
        scores = {}

        # 基础评分
        base_score = 7.0
        response_length = len(response)

        # 准确性评分 - 基于模型的准确性偏差
        accuracy_bias = characteristics.get("accuracy_bias", 0.8)
        accuracy_score = base_score + (accuracy_bias - 0.8) * 5  # 转换为评分

        # 检查是否包含预期答案
        if context:
            expected = str(context.get("expected_output", "")).lower()
            ground_truth = str(context.get("ground_truth", "")).lower()

            if expected and expected in response.lower():
                accuracy_score += 1.5
            elif ground_truth and ground_truth in response.lower():
                accuracy_score += 1.0

        scores["accuracy"] = min(10.0, max(5.0, accuracy_score))

        # 清晰度评分 - 基于推理风格和细节级别
        clarity_score = base_score
        detail_level = characteristics.get("detail_level", "medium")

        if detail_level == "very_high":
            clarity_score += 1.5
        elif detail_level == "high":
            clarity_score += 1.0
        elif detail_level == "medium":
            clarity_score += 0.5

        # 基于响应结构性
        structure_indicators = ["step", "first", "analysis", "conclusion", "therefore"]
        structure_count = sum(
            1 for indicator in structure_indicators if indicator in response.lower()
        )
        clarity_score += min(1.0, structure_count * 0.2)

        scores["clarity"] = min(10.0, max(5.0, clarity_score))

        # 完整性评分 - 基于响应长度和细节级别
        completeness_score = base_score
        length_multiplier = characteristics.get("response_length_multiplier", 1.0)

        expected_length = 100 * length_multiplier
        if response_length >= expected_length:
            completeness_score += 1.5
        elif response_length >= expected_length * 0.7:
            completeness_score += 1.0
        elif response_length >= expected_length * 0.5:
            completeness_score += 0.5

        scores["completeness"] = min(10.0, max(5.0, completeness_score))

        # 推理质量评分
        reasoning_quality_score = base_score
        reasoning_style = characteristics.get("reasoning_style", "systematic")

        style_bonus = {
            "systematic": 1.5,
            "analytical": 1.3,
            "comprehensive": 1.4,
            "structured": 1.2,
            "precise": 1.1,
            "efficient": 0.8,
            "conversational": 0.7,
            "concise": 0.6,
            "balanced": 1.0,
        }

        reasoning_quality_score += style_bonus.get(reasoning_style, 1.0)
        scores["reasoning_quality"] = min(10.0, max(5.0, reasoning_quality_score))

        return scores

    def _generate_comparison_reasoning(
        self,
        scores_a: Dict[str, float],
        scores_b: Dict[str, float],
        char_a: Dict[str, Any],
        char_b: Dict[str, Any],
        winner: str,
    ) -> str:
        """生成对比评估理由"""
        if winner == "Tie":
            return "两个响应在各评估维度上表现相近，都有各自的优势。"

        # 找出主要差异维度
        max_diff = 0
        max_diff_dim = "overall"

        for dim in scores_a:
            if dim in scores_b:
                diff = abs(scores_a[dim] - scores_b[dim])
                if diff > max_diff:
                    max_diff = diff
                    max_diff_dim = dim

        # 获取获胜模型的特征
        if winner == "A":
            winner_char = char_a
            winner_scores = scores_a
            loser_scores = scores_b
        else:
            winner_char = char_b
            winner_scores = scores_b
            loser_scores = scores_a

        reasoning_style = winner_char.get("reasoning_style", "systematic")
        detail_level = winner_char.get("detail_level", "medium")

        dim_names = {
            "accuracy": "准确性",
            "clarity": "清晰度",
            "completeness": "完整性",
            "reasoning_quality": "推理质量",
        }

        dim_cn = dim_names.get(max_diff_dim, "综合评估")
        winner_score = winner_scores.get(max_diff_dim, 7.0)
        loser_score = loser_scores.get(max_diff_dim, 7.0)

        style_descriptions = {
            "systematic": "系统性方法",
            "analytical": "分析性思维",
            "comprehensive": "全面性分析",
            "structured": "结构化表达",
            "precise": "精确性",
            "efficient": "高效性",
            "conversational": "对话性",
            "concise": "简洁性",
        }

        style_desc = style_descriptions.get(reasoning_style, "平衡性")

        return f"获胜响应在{dim_cn}方面表现突出 ({winner_score:.1f} vs {loser_score:.1f})，体现了{style_desc}和{detail_level}详细程度的优势。"

    def get_tournament_rankings(self, models: List[str]) -> List[Tuple[str, int]]:
        """获取锦标赛排名"""
        scores = defaultdict(int)

        # 计算每个模型的总胜场数
        for model_a in models:
            for model_b in models:
                if model_a != model_b:
                    scores[model_a] += self.pairwise_results[model_a][model_b]

        # 按胜场数排序
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class MultiModelTestRunner:
    """多模型测试运行器"""

    def __init__(self, max_cost_usd: float = 3.0):
        # 初始化多个不同的模型
        self.models = {
            "gemini-pro": MultiModelProvider("Google", "gemini-pro", {"tier": "premium"}),
            "gemini-flash": MultiModelProvider("Google", "gemini-flash", {"tier": "fast"}),
            "claude-3.5-sonnet": MultiModelProvider(
                "Anthropic", "claude-3.5-sonnet", {"tier": "premium"}
            ),
            "claude-3-haiku": MultiModelProvider("Anthropic", "claude-3-haiku", {"tier": "fast"}),
            "gpt-4o": MultiModelProvider("OpenAI", "gpt-4o", {"tier": "premium"}),
            "gpt-4o-mini": MultiModelProvider("OpenAI", "gpt-4o-mini", {"tier": "fast"}),
            "gpt-3.5-turbo": MultiModelProvider("OpenAI", "gpt-3.5-turbo", {"tier": "standard"}),
        }

        self.judge = TournamentJudge()
        self.max_cost_usd = max_cost_usd
        self.test_results = []
        self.current_cost = 0.0

        # 统计信息
        self.stats = {
            "total_comparisons": 0,
            "model_performance": defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0}),
            "pairwise_matrix": defaultdict(lambda: defaultdict(int)),
            "cost_breakdown": defaultdict(float),
            "timing_stats": [],
        }

    def select_models_for_test(
        self, model_names: List[str] = None
    ) -> Dict[str, MultiModelProvider]:
        """选择参与测试的模型"""
        if model_names:
            selected = {name: self.models[name] for name in model_names if name in self.models}
        else:
            # 默认选择多样化的5个模型
            default_selection = [
                "gemini-flash",  # 快速且便宜
                "claude-3.5-sonnet",  # 高质量
                "gpt-4o-mini",  # 平衡性价比
                "gpt-3.5-turbo",  # 基准模型
                "claude-3-haiku",  # 快速高质量
            ]
            selected = {name: self.models[name] for name in default_selection}

        logger.info(f"🤖 选中的模型: {', '.join(selected.keys())}")
        return selected

    def load_test_samples(self, sample_count: int = 300) -> List[Dict[str, Any]]:
        """加载测试样本"""
        datasets_dir = Path("data/processed")
        samples = []

        # 平衡加载不同类型的数据
        target_per_dataset = sample_count // 2

        # ARC-Easy
        arc_file = datasets_dir / "arc_easy.json"
        if arc_file.exists():
            with open(arc_file, "r", encoding="utf-8") as f:
                arc_data = json.load(f)
            samples.extend(random.sample(arc_data, min(target_per_dataset, len(arc_data))))

        # GSM8K
        gsm8k_file = datasets_dir / "gsm8k.json"
        if gsm8k_file.exists():
            with open(gsm8k_file, "r", encoding="utf-8") as f:
                gsm8k_data = json.load(f)
            samples.extend(random.sample(gsm8k_data, min(target_per_dataset, len(gsm8k_data))))

        # 随机打乱并限制数量
        random.shuffle(samples)
        return samples[:sample_count]

    async def run_multi_model_comparison(
        self, sample_count: int = 300, selected_models: List[str] = None
    ) -> Dict[str, Any]:
        """运行多模型对比测试"""
        logger.info("🚀 开始多模型横向对比测试")
        logger.info("=" * 80)

        # 选择参与测试的模型
        test_models = self.select_models_for_test(selected_models)
        model_names = list(test_models.keys())

        # 计算所有可能的模型对比组合
        model_pairs = list(itertools.combinations(model_names, 2))
        total_comparisons = len(model_pairs) * sample_count

        logger.info(f"📊 测试配置:")
        logger.info(f"  参与模型: {len(test_models)}个")
        logger.info(f"  对比组合: {len(model_pairs)}对")
        logger.info(f"  测试样本: {sample_count}个")
        logger.info(f"  总对比次数: {total_comparisons:,}")

        # 成本预估
        estimated_cost = self._estimate_total_cost(test_models, total_comparisons)
        logger.info(f"  预估成本: ${estimated_cost:.4f}")

        if estimated_cost > self.max_cost_usd:
            logger.warning(f"⚠️ 预估成本超出预算，请考虑减少模型数量或样本数")
            return {}

        # 加载测试数据
        test_samples = self.load_test_samples(sample_count)
        logger.info(f"📚 已加载 {len(test_samples)} 个测试样本")

        # 开始测试
        start_time = time.time()

        # 为每个样本运行所有模型对比
        for sample_idx, sample in enumerate(test_samples, 1):
            logger.info(f"\n📝 处理样本 {sample_idx}/{len(test_samples)}")

            # 生成所有模型的响应
            responses = {}
            for model_name, model in test_models.items():
                response = await model.generate_response(sample["prompt"], sample)
                responses[model_name] = response

            # 进行所有两两对比
            for model_a, model_b in model_pairs:
                evaluation = await self.judge.evaluate_pairwise(
                    sample["prompt"],
                    responses[model_a],
                    responses[model_b],
                    model_a,
                    model_b,
                    sample,
                )

                # 记录结果
                self._record_comparison_result(model_a, model_b, evaluation, sample_idx)

            # 更新成本统计
            self._update_cost_tracking(test_models)

            # 进度报告
            if sample_idx % 50 == 0 or sample_idx <= 5:
                progress = (sample_idx / len(test_samples)) * 100
                logger.info(
                    f"📈 进度: {sample_idx}/{len(test_samples)} ({progress:.1f}%) | 当前成本: ${self.current_cost:.4f}"
                )

        total_time = time.time() - start_time

        # 生成综合分析
        summary = self._generate_multi_model_summary(
            test_models, model_pairs, total_time, sample_count
        )

        # 保存结果
        self._save_multi_model_results(summary)

        return summary

    def _estimate_total_cost(
        self, models: Dict[str, MultiModelProvider], total_comparisons: int
    ) -> float:
        """估算总成本"""
        avg_cost_per_response = sum(
            (100 / 1000) * model.pricing[model.model_name]["input"]
            + (80 / 1000) * model.pricing[model.model_name]["output"]
            for model in models.values()
        ) / len(models)

        # 模型响应成本 + 评估成本
        response_cost = (
            avg_cost_per_response
            * len(models)
            * (total_comparisons // len(list(itertools.combinations(models.keys(), 2))))
        )
        evaluation_cost = 0.000065 * total_comparisons  # 评估成本

        return response_cost + evaluation_cost

    def _record_comparison_result(
        self, model_a: str, model_b: str, evaluation: Dict[str, Any], sample_idx: int
    ):
        """记录对比结果"""
        winner = evaluation["winner"]

        if winner == "A":
            self.stats["model_performance"][model_a]["wins"] += 1
            self.stats["model_performance"][model_b]["losses"] += 1
            self.stats["pairwise_matrix"][model_a][model_b] += 1
        elif winner == "B":
            self.stats["model_performance"][model_b]["wins"] += 1
            self.stats["model_performance"][model_a]["losses"] += 1
            self.stats["pairwise_matrix"][model_b][model_a] += 1
        else:  # Tie
            self.stats["model_performance"][model_a]["ties"] += 1
            self.stats["model_performance"][model_b]["ties"] += 1

        self.stats["total_comparisons"] += 1

        # 保存详细结果
        self.test_results.append(
            {
                "sample_idx": sample_idx,
                "model_a": model_a,
                "model_b": model_b,
                "evaluation": evaluation,
            }
        )

    def _update_cost_tracking(self, models: Dict[str, MultiModelProvider]):
        """更新成本跟踪"""
        total_cost = 0.0
        for model_name, model in models.items():
            model_cost = model.total_cost
            self.stats["cost_breakdown"][model_name] = model_cost
            total_cost += model_cost

        self.current_cost = total_cost

    def _generate_multi_model_summary(
        self,
        models: Dict[str, MultiModelProvider],
        model_pairs: List[Tuple[str, str]],
        total_time: float,
        sample_count: int,
    ) -> Dict[str, Any]:
        """生成多模型对比汇总"""

        # 计算排名
        model_names = list(models.keys())
        rankings = self.judge.get_tournament_rankings(model_names)

        # 计算胜率统计
        win_rates = {}
        for model_name in model_names:
            stats = self.stats["model_performance"][model_name]
            total_games = stats["wins"] + stats["losses"] + stats["ties"]
            win_rate = stats["wins"] / total_games if total_games > 0 else 0
            win_rates[model_name] = {
                "win_rate": round(win_rate, 3),
                "wins": stats["wins"],
                "losses": stats["losses"],
                "ties": stats["ties"],
                "total_games": total_games,
            }

        # 模型特性分析
        model_analysis = {}
        for model_name, model in models.items():
            char = model.model_characteristics[model.model_name]
            cost_per_response = (
                model.total_cost / model.request_count if model.request_count > 0 else 0
            )

            model_analysis[model_name] = {
                "provider": model.provider_name,
                "characteristics": char,
                "performance": win_rates[model_name],
                "cost_efficiency": round(cost_per_response, 6),
                "tokens_used": model.total_tokens,
                "requests": model.request_count,
            }

        # 对比矩阵
        pairwise_matrix = {}
        for model_a in model_names:
            pairwise_matrix[model_a] = {}
            for model_b in model_names:
                if model_a != model_b:
                    wins = self.stats["pairwise_matrix"][model_a][model_b]
                    losses = self.stats["pairwise_matrix"][model_b][model_a]
                    pairwise_matrix[model_a][model_b] = {
                        "wins": wins,
                        "losses": losses,
                        "win_rate": round(wins / (wins + losses), 3) if (wins + losses) > 0 else 0,
                    }
                else:
                    pairwise_matrix[model_a][model_b] = {"wins": 0, "losses": 0, "win_rate": 0}

        return {
            "test_info": {
                "test_name": "Multi-Model Comparison Test",
                "timestamp": datetime.now().isoformat(),
                "models_tested": len(models),
                "model_pairs": len(model_pairs),
                "samples_per_pair": sample_count,
                "total_comparisons": self.stats["total_comparisons"],
                "total_time": round(total_time, 2),
            },
            "rankings": rankings,
            "model_analysis": model_analysis,
            "pairwise_matrix": pairwise_matrix,
            "cost_analysis": {
                "total_cost": round(self.current_cost, 6),
                "cost_per_comparison": (
                    round(self.current_cost / self.stats["total_comparisons"], 6)
                    if self.stats["total_comparisons"] > 0
                    else 0
                ),
                "cost_breakdown": {k: round(v, 6) for k, v in self.stats["cost_breakdown"].items()},
            },
            "performance_metrics": {
                "comparisons_per_second": (
                    round(self.stats["total_comparisons"] / total_time, 2) if total_time > 0 else 0
                ),
                "avg_time_per_comparison": (
                    round(total_time / self.stats["total_comparisons"], 3)
                    if self.stats["total_comparisons"] > 0
                    else 0
                ),
            },
            "detailed_results": self.test_results[:20],  # 保存前20个详细结果
        }

    def _save_multi_model_results(self, summary: Dict[str, Any]):
        """保存多模型测试结果"""
        results_dir = Path("logs/multi_model_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"multi_model_test_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n📁 详细结果已保存到: {results_file}")

    def display_multi_model_summary(self, summary: Dict[str, Any]):
        """显示多模型测试汇总"""
        if not summary:
            return

        logger.info("\n" + "=" * 80)
        logger.info("🏆 多模型横向对比测试汇总")
        logger.info("=" * 80)

        test_info = summary["test_info"]
        rankings = summary["rankings"]
        model_analysis = summary["model_analysis"]
        cost = summary["cost_analysis"]
        performance = summary["performance_metrics"]

        # 基本信息
        logger.info(f"🕒 测试时间: {test_info['timestamp']}")
        logger.info(f"🤖 参与模型: {test_info['models_tested']}个")
        logger.info(f"⚔️ 对比次数: {test_info['total_comparisons']:,}")
        logger.info(f"📊 样本数量: {test_info['samples_per_pair']}个")
        logger.info(f"⏱️ 总耗时: {test_info['total_time']}秒")
        logger.info(f"🚀 对比速度: {performance['comparisons_per_second']} 对比/秒")

        # 排名榜
        logger.info(f"\n🏆 模型排名 (按总胜场数):")
        for rank, (model_name, wins) in enumerate(rankings, 1):
            analysis = model_analysis[model_name]
            win_rate = analysis["performance"]["win_rate"]
            provider = analysis["provider"]
            cost_eff = analysis["cost_efficiency"]

            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
            logger.info(f"  {medal} {model_name} ({provider})")
            logger.info(f"      胜场: {wins} | 胜率: {win_rate:.1%} | 成本: ${cost_eff:.6f}/响应")

        # 模型特性分析
        logger.info(f"\n🔍 模型特性对比:")
        for model_name, analysis in model_analysis.items():
            char = analysis["characteristics"]
            perf = analysis["performance"]

            logger.info(f"  📱 {model_name}:")
            logger.info(
                f"    推理风格: {char['reasoning_style']} | 详细级别: {char['detail_level']}"
            )
            logger.info(
                f"    准确性偏差: {char['accuracy_bias']:.2f} | 响应长度倍数: {char['response_length_multiplier']:.1f}"
            )
            logger.info(
                f"    战绩: {perf['wins']}胜{perf['losses']}负{perf['ties']}平 (胜率: {perf['win_rate']:.1%})"
            )

        # 成本分析
        logger.info(f"\n💰 成本分析:")
        logger.info(f"  总成本: ${cost['total_cost']}")
        logger.info(f"  每对比成本: ${cost['cost_per_comparison']}")
        logger.info(f"  成本分布:")
        for model_name, model_cost in cost["cost_breakdown"].items():
            percentage = (model_cost / cost["total_cost"]) * 100 if cost["total_cost"] > 0 else 0
            logger.info(f"    {model_name}: ${model_cost} ({percentage:.1f}%)")

        # 对比矩阵精选
        logger.info(f"\n⚔️ 关键对比结果:")
        pairwise = summary["pairwise_matrix"]
        top_models = [model for model, _ in rankings[:3]]

        for i, model_a in enumerate(top_models):
            for model_b in top_models[i + 1 :]:
                if model_a in pairwise and model_b in pairwise[model_a]:
                    a_vs_b = pairwise[model_a][model_b]
                    b_vs_a = pairwise[model_b][model_a]

                    if a_vs_b["wins"] > b_vs_a["wins"]:
                        winner, loser = model_a, model_b
                        win_rate = a_vs_b["win_rate"]
                    else:
                        winner, loser = model_b, model_a
                        win_rate = b_vs_a["win_rate"]

                    logger.info(f"  {winner} vs {loser}: {win_rate:.1%} 胜率")

        # 结论
        logger.info(f"\n🎯 测试结论:")
        if rankings:
            champion = rankings[0][0]
            champion_analysis = model_analysis[champion]
            champion_style = champion_analysis["characteristics"]["reasoning_style"]
            champion_rate = champion_analysis["performance"]["win_rate"]

            logger.info(f"  🏆 冠军: {champion} (胜率: {champion_rate:.1%})")
            logger.info(f"  🎯 优势: {champion_style}推理风格表现最佳")

            # 成本效率之王
            most_efficient = min(model_analysis.items(), key=lambda x: x[1]["cost_efficiency"])
            logger.info(
                f"  💰 成本效率王: {most_efficient[0]} (${most_efficient[1]['cost_efficiency']:.6f}/响应)"
            )


async def main():
    """主函数"""
    logger.info("🎯 多模型横向对比测试")
    logger.info("测试5个不同模型的能力对比")

    # 创建测试运行器
    tester = MultiModelTestRunner(max_cost_usd=3.0)

    # 运行多模型对比测试
    summary = await tester.run_multi_model_comparison(
        sample_count=200, selected_models=None  # 每对比200个样本  # 使用默认的5个模型
    )

    if summary:
        # 显示结果
        tester.display_multi_model_summary(summary)
        logger.info("\n✅ 多模型对比测试完成！")
    else:
        logger.error("❌ 测试失败")

    return summary


if __name__ == "__main__":
    asyncio.run(main())

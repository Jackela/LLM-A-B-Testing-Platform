#!/usr/bin/env python3
"""
Cost-Controlled Large Scale Evaluation Test
成本控制的大规模评估测试 - 模拟真实API调用和成本分析
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


class CostControlledProvider:
    """成本控制的LLM提供商模拟器"""

    def __init__(self, provider_name: str, model_name: str, pricing_tier: str = "budget"):
        self.provider_name = provider_name
        self.model_name = model_name
        self.pricing_tier = pricing_tier
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # 真实价格表 (每1K tokens, USD)
        self.pricing = {
            "gemini-flash": {"input": 0.000075, "output": 0.0003},  # 超便宜
            "claude-sonnet": {"input": 0.003, "output": 0.015},  # 中等
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # 便宜
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},  # 很便宜
        }

        # 模型特性配置
        self.model_characteristics = {
            "gemini-flash": {
                "response_style": "structured",
                "avg_response_length": 120,
                "detail_level": "medium",
                "reasoning_depth": "moderate",
            },
            "claude-sonnet": {
                "response_style": "analytical",
                "avg_response_length": 150,
                "detail_level": "high",
                "reasoning_depth": "deep",
            },
            "gpt-3.5-turbo": {
                "response_style": "conversational",
                "avg_response_length": 100,
                "detail_level": "medium",
                "reasoning_depth": "moderate",
            },
            "gpt-4o-mini": {
                "response_style": "precise",
                "avg_response_length": 90,
                "detail_level": "focused",
                "reasoning_depth": "efficient",
            },
        }

    async def generate_response(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """生成响应并计算精确成本"""

        # 模拟API延迟
        await asyncio.sleep(random.uniform(0.2, 0.8))

        # 获取模型特性
        char = self.model_characteristics.get(
            self.model_name, self.model_characteristics["gpt-3.5-turbo"]
        )

        # 生成响应内容
        content = self._generate_realistic_response(prompt, context, char)

        # 计算token使用量
        input_tokens = self._calculate_tokens(prompt)
        output_tokens = self._calculate_tokens(content)

        # 计算精确成本
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
        }

    def _generate_realistic_response(
        self, prompt: str, context: Dict[str, Any], char: Dict[str, str]
    ) -> str:
        """基于模型特性生成真实的响应"""

        if context and context.get("source") == "ARC-Easy":
            return self._generate_science_response(prompt, context, char)
        elif context and context.get("source") == "GSM8K":
            return self._generate_math_response(prompt, context, char)
        else:
            return self._generate_general_response(prompt, char)

    def _generate_science_response(
        self, prompt: str, context: Dict[str, Any], char: Dict[str, str]
    ) -> str:
        """生成科学问题响应"""
        expected = context.get("expected_output", "scientific answer")

        if char["response_style"] == "structured":
            return f"Based on scientific principles: {expected}. This follows from fundamental concepts in the field."
        elif char["response_style"] == "analytical":
            return f"Analyzing this scientifically, we can determine that {expected}. The reasoning involves understanding the underlying mechanisms and applying established scientific knowledge to reach this conclusion."
        elif char["response_style"] == "conversational":
            return f"So looking at this science question, the answer is {expected}. This makes sense when you think about how these processes work."
        else:  # precise
            return f"Answer: {expected}. Scientific basis: established principles."

    def _generate_math_response(
        self, prompt: str, context: Dict[str, Any], char: Dict[str, str]
    ) -> str:
        """生成数学问题响应"""
        answer = context.get("ground_truth", "calculated result")

        if char["response_style"] == "structured":
            return f"Step-by-step solution: First, identify given values. Then apply appropriate operations. Final answer: {answer}."
        elif char["response_style"] == "analytical":
            return f"To solve this problem, I need to carefully analyze the given information and apply mathematical reasoning. Working through the calculations systematically, the answer is {answer}."
        elif char["response_style"] == "conversational":
            return (
                f"Let me work through this math problem. When I calculate it out, I get {answer}."
            )
        else:  # precise
            return f"Calculation: {answer}."

    def _generate_general_response(self, prompt: str, char: Dict[str, str]) -> str:
        """生成通用响应"""
        if char["response_style"] == "structured":
            return "Based on the information provided, I can offer a structured analysis of this topic with relevant details."
        elif char["response_style"] == "analytical":
            return "This question requires careful consideration of multiple factors. Let me provide a comprehensive analysis that addresses the key aspects."
        elif char["response_style"] == "conversational":
            return "That's an interesting question! Let me share some thoughts on this topic."
        else:  # precise
            return "Response: Direct answer addressing the core question."

    def _calculate_tokens(self, text: str) -> int:
        """计算token数量（近似）"""
        # 简化的token计算：约1.3倍单词数
        word_count = len(text.split())
        return max(1, int(word_count * 1.3))

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """计算精确成本"""
        pricing = self.pricing.get(self.model_name, self.pricing["gpt-3.5-turbo"])

        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost


class RealisticLLMJudge:
    """真实的LLM评判者"""

    def __init__(self):
        self.evaluation_count = 0
        self.judge_provider = CostControlledProvider("Judge", "gpt-4o-mini", "budget")

    async def evaluate_responses(
        self,
        question: str,
        response_a: Dict[str, Any],
        response_b: Dict[str, Any],
        provider_a: str,
        provider_b: str,
        context: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """评估两个响应"""

        # 构建评估提示
        judge_prompt = self._build_evaluation_prompt(
            question, response_a["content"], response_b["content"], context
        )

        # 使用Judge模型进行评估
        judge_response = await self.judge_provider.generate_response(judge_prompt)

        # 解析评估结果
        evaluation = self._parse_evaluation(
            judge_response["content"], response_a["content"], response_b["content"], context
        )
        evaluation["judge_cost"] = judge_response["cost"]
        evaluation["judge_tokens"] = (
            judge_response["input_tokens"] + judge_response["output_tokens"]
        )

        self.evaluation_count += 1
        return evaluation

    def _build_evaluation_prompt(
        self, question: str, response_a: str, response_b: str, context: Dict[str, Any]
    ) -> str:
        """构建评估提示"""
        category = context.get("category", "general") if context else "general"

        return f"""
Evaluate which response is better for this {category} question.

Question: {question}

Response A: {response_a}

Response B: {response_b}

Consider: accuracy, clarity, completeness, helpfulness.
Choose: A, B, or Tie
Provide brief reasoning.
"""

    def _parse_evaluation(
        self, judge_response: str, response_a: str, response_b: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """解析评估结果"""

        # 启发式评估逻辑
        len_a, len_b = len(response_a), len(response_b)

        # 基础评分
        score_a = 7.0
        score_b = 7.0

        # 检查是否包含预期答案
        if context:
            expected = str(context.get("expected_output", "")).lower()
            ground_truth = str(context.get("ground_truth", "")).lower()

            if expected and (expected in response_a.lower() or ground_truth in response_a.lower()):
                score_a += 1.5
            if expected and (expected in response_b.lower() or ground_truth in response_b.lower()):
                score_b += 1.5

        # 长度和详细程度
        if len_a > len_b * 1.2:
            score_a += 0.5
        elif len_b > len_a * 1.2:
            score_b += 0.5

        # 结构化表达
        if any(word in response_a.lower() for word in ["step", "first", "analysis", "because"]):
            score_a += 0.5
        if any(word in response_b.lower() for word in ["step", "first", "analysis", "because"]):
            score_b += 0.5

        # 确定获胜者
        score_diff = abs(score_a - score_b)
        if score_a > score_b + 0.3:
            winner = "A"
            confidence = min(0.95, 0.6 + score_diff / 5)
        elif score_b > score_a + 0.3:
            winner = "B"
            confidence = min(0.95, 0.6 + score_diff / 5)
        else:
            winner = "Tie"
            confidence = 0.5

        # 生成评估理由
        if winner == "A":
            reasoning = f"Response A scores higher ({score_a:.1f} vs {score_b:.1f}) due to better accuracy or detail."
        elif winner == "B":
            reasoning = f"Response B scores higher ({score_b:.1f} vs {score_a:.1f}) due to better accuracy or detail."
        else:
            reasoning = (
                f"Both responses are comparable in quality ({score_a:.1f} vs {score_b:.1f})."
            )

        return {
            "winner": winner,
            "confidence": round(confidence, 2),
            "reasoning": reasoning,
            "scores": {"response_a": round(score_a, 1), "response_b": round(score_b, 1)},
        }


class CostControlledTestRunner:
    """成本控制的测试运行器 - 性能优化版"""

    def __init__(self, max_cost_usd: float = 2.0):
        # 初始化便宜的模型
        self.provider_a = CostControlledProvider("Gemini", "gemini-flash")
        self.provider_b = CostControlledProvider("GPT-4o", "gpt-4o-mini")
        self.judge = RealisticLLMJudge()

        self.max_cost_usd = max_cost_usd
        self.test_results = []
        self.current_cost = 0.0

        # 性能优化: 数据预加载和缓存
        self._cached_samples = None
        self._connection_pool = None
        self._batch_size = 10  # 增加批处理大小
        self._metrics_cache = {}

        # 资源管理优化
        self._semaphore = asyncio.Semaphore(50)  # 限制并发数避免资源耗尽
        self._start_time = None

    def load_test_samples(self, target_count: int = 100) -> List[Dict[str, Any]]:
        """加载测试样本 - 性能优化版本"""
        # 缓存样本避免重复加载
        if self._cached_samples and len(self._cached_samples) >= target_count:
            return self._cached_samples[:target_count]

        datasets_dir = Path("data/processed")
        all_samples = []

        # 并行加载数据集以提高效率
        async def load_dataset_async():
            tasks = []

            # 加载ARC数据集
            arc_file = datasets_dir / "arc_easy.json"
            if arc_file.exists():
                tasks.append(self._load_json_file(arc_file))

            # 加载GSM8K数据集
            gsm8k_file = datasets_dir / "gsm8k.json"
            if gsm8k_file.exists():
                tasks.append(self._load_json_file(gsm8k_file))

            if tasks:
                datasets = await asyncio.gather(*tasks)
                for dataset in datasets:
                    all_samples.extend(dataset[: target_count // len(datasets)])

        # 兼容同步调用的临时解决方案
        try:
            asyncio.get_event_loop().run_until_complete(load_dataset_async())
        except RuntimeError:
            # 回退到同步加载
            arc_file = datasets_dir / "arc_easy.json"
            if arc_file.exists():
                with open(arc_file, "r", encoding="utf-8") as f:
                    arc_data = json.load(f)
                all_samples.extend(arc_data[: target_count // 2])

            gsm8k_file = datasets_dir / "gsm8k.json"
            if gsm8k_file.exists():
                with open(gsm8k_file, "r", encoding="utf-8") as f:
                    gsm8k_data = json.load(f)
                all_samples.extend(gsm8k_data[: target_count // 2])

        # 预处理和缓存
        random.shuffle(all_samples)
        self._cached_samples = all_samples[: target_count * 2]  # 缓存更多样本供后续使用
        return self._cached_samples[:target_count]

    async def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """异步加载JSON文件"""
        import aiofiles

        try:
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content)
        except Exception:
            # 回退到同步加载
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)

    def estimate_cost_per_sample(self) -> float:
        """估算每个样本的成本"""
        # 基于定价计算预估成本
        avg_input_tokens = 100  # 平均输入token数
        avg_output_tokens = 80  # 平均输出token数

        # Provider A成本
        pricing_a = self.provider_a.pricing[self.provider_a.model_name]
        cost_a = (avg_input_tokens / 1000) * pricing_a["input"] + (
            avg_output_tokens / 1000
        ) * pricing_a["output"]

        # Provider B成本
        pricing_b = self.provider_b.pricing[self.provider_b.model_name]
        cost_b = (avg_input_tokens / 1000) * pricing_b["input"] + (
            avg_output_tokens / 1000
        ) * pricing_b["output"]

        # Judge成本
        judge_input = 200  # 评估提示更长
        judge_output = 50  # 评估结果较短
        pricing_judge = self.judge.judge_provider.pricing[self.judge.judge_provider.model_name]
        cost_judge = (judge_input / 1000) * pricing_judge["input"] + (
            judge_output / 1000
        ) * pricing_judge["output"]

        return cost_a + cost_b + cost_judge

    async def run_cost_controlled_test(self, target_samples: int = 100) -> Dict[str, Any]:
        """运行成本控制的测试"""
        logger.info("🚀 开始成本控制的大规模LLM评估测试")
        logger.info("=" * 70)

        # 成本预估
        estimated_cost_per_sample = self.estimate_cost_per_sample()
        estimated_total_cost = estimated_cost_per_sample * target_samples

        logger.info(f"💰 成本预估:")
        logger.info(f"  每样本预估成本: ${estimated_cost_per_sample:.6f}")
        logger.info(f"  {target_samples}样本预估总成本: ${estimated_total_cost:.4f}")
        logger.info(f"  预算上限: ${self.max_cost_usd:.2f}")

        if estimated_total_cost > self.max_cost_usd:
            max_safe_samples = int(self.max_cost_usd / estimated_cost_per_sample)
            logger.warning(f"⚠️ 预估成本超出预算，建议限制为 {max_safe_samples} 样本")
            target_samples = min(target_samples, max_safe_samples)

        # 加载测试数据
        test_samples = self.load_test_samples(target_samples)
        actual_samples = len(test_samples)

        logger.info(f"📚 已加载 {actual_samples} 个测试样本")
        logger.info(
            f"🤖 Provider A: {self.provider_a.provider_name} ({self.provider_a.model_name})"
        )
        logger.info(
            f"🤖 Provider B: {self.provider_b.provider_name} ({self.provider_b.model_name})"
        )
        logger.info(
            f"⚖️ Judge: {self.judge.judge_provider.provider_name} ({self.judge.judge_provider.model_name})"
        )

        start_time = time.time()
        completed_samples = 0

        # 批量并发处理样本，提高测试效率
        batch_size = min(self._batch_size, len(test_samples))
        batches = [
            test_samples[i : i + batch_size] for i in range(0, len(test_samples), batch_size)
        ]

        for batch_idx, batch in enumerate(batches):
            # 检查成本限制
            if self.current_cost >= self.max_cost_usd:
                logger.warning(f"💰 达到成本上限 ${self.max_cost_usd:.2f}，停止测试")
                break

            try:
                # 并发处理批次
                batch_tasks = []
                for i, sample in enumerate(batch):
                    sample_idx = batch_idx * batch_size + i + 1
                    task = self._process_sample_with_semaphore(sample, sample_idx)
                    batch_tasks.append(task)

                # 等待批次完成
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                # 处理批次结果
                for result in batch_results:
                    if isinstance(result, Exception):
                        logger.error(f"❌ 批次处理失败: {result}")
                        continue

                    if result:  # 确保结果有效
                        self.test_results.append(result)
                        completed_samples += 1

                # 更新当前成本
                self.current_cost = (
                    self.provider_a.total_cost
                    + self.provider_b.total_cost
                    + self.judge.judge_provider.total_cost
                )

                # 进度报告
                progress = (completed_samples / actual_samples) * 100
                logger.info(
                    f"📊 批次 {batch_idx + 1}/{len(batches)} 完成 | 进度: {completed_samples}/{actual_samples} ({progress:.1f}%) | 当前成本: ${self.current_cost:.4f}"
                )

                # 批次间短暂延迟，避免过度负载
                if batch_idx < len(batches) - 1:
                    await asyncio.sleep(0.1)

            except Exception as e:
                logger.error(f"❌ 批次 {batch_idx + 1} 处理失败: {e}")
                continue

        total_time = time.time() - start_time

        # 生成汇总
        summary = self._generate_cost_summary(total_time, target_samples, completed_samples)

        # 保存结果
        self._save_cost_results(summary)

        return summary

    async def _process_sample_with_semaphore(
        self, sample: Dict[str, Any], sample_num: int
    ) -> Optional[Dict[str, Any]]:
        """使用信号量控制并发的样本处理"""
        async with self._semaphore:
            try:
                return await self.evaluate_single_sample(sample, sample_num)
            except Exception as e:
                logger.error(f"❌ 样本 {sample_num} 处理失败: {e}")
                return None

    async def evaluate_single_sample(
        self, sample: Dict[str, Any], sample_num: int
    ) -> Dict[str, Any]:
        """评估单个样本"""
        question = sample["prompt"]

        # 并发获取响应
        response_a_task = self.provider_a.generate_response(question, sample)
        response_b_task = self.provider_b.generate_response(question, sample)

        start_time = time.time()
        response_a, response_b = await asyncio.gather(response_a_task, response_b_task)
        response_time = time.time() - start_time

        # 评估
        eval_start_time = time.time()
        evaluation = await self.judge.evaluate_responses(
            question,
            response_a,
            response_b,
            self.provider_a.provider_name,
            self.provider_b.provider_name,
            sample,
        )
        eval_time = time.time() - eval_start_time

        # 计算总成本
        total_cost = response_a["cost"] + response_b["cost"] + evaluation["judge_cost"]

        return {
            "sample_num": sample_num,
            "sample_data": {
                "id": sample["id"],
                "category": sample.get("category", "unknown"),
                "source": sample.get("source", "unknown"),
            },
            "responses": {
                "provider_a": {
                    "name": f"{response_a['provider']} ({response_a['model_name']})",
                    "content": (
                        response_a["content"][:150] + "..."
                        if len(response_a["content"]) > 150
                        else response_a["content"]
                    ),
                    "tokens": response_a["input_tokens"] + response_a["output_tokens"],
                    "cost": response_a["cost"],
                },
                "provider_b": {
                    "name": f"{response_b['provider']} ({response_b['model_name']})",
                    "content": (
                        response_b["content"][:150] + "..."
                        if len(response_b["content"]) > 150
                        else response_b["content"]
                    ),
                    "tokens": response_b["input_tokens"] + response_b["output_tokens"],
                    "cost": response_b["cost"],
                },
            },
            "evaluation": evaluation,
            "timing": {
                "response_time": round(response_time, 3),
                "evaluation_time": round(eval_time, 3),
            },
            "total_cost": round(total_cost, 6),
        }

    def _generate_cost_summary(
        self, total_time: float, target_samples: int, completed_samples: int
    ) -> Dict[str, Any]:
        """生成成本汇总"""
        if not self.test_results:
            return {}

        # 基础统计
        provider_a_wins = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "A")
        provider_b_wins = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "B")
        ties = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "Tie")

        # 成本分析
        total_cost = self.current_cost
        avg_cost_per_sample = total_cost / completed_samples if completed_samples > 0 else 0

        provider_a_cost = self.provider_a.total_cost
        provider_b_cost = self.provider_b.total_cost
        judge_cost = self.judge.judge_provider.total_cost

        # Token分析
        total_tokens = (
            self.provider_a.total_tokens
            + self.provider_b.total_tokens
            + self.judge.judge_provider.total_tokens
        )

        # 性能分析
        avg_response_time = (
            sum(r["timing"]["response_time"] for r in self.test_results) / completed_samples
        )
        avg_eval_time = (
            sum(r["timing"]["evaluation_time"] for r in self.test_results) / completed_samples
        )

        # 按类型分析
        category_analysis = {}
        for result in self.test_results:
            category = result["sample_data"]["category"]
            if category not in category_analysis:
                category_analysis[category] = {"A": 0, "B": 0, "Tie": 0, "total": 0}

            winner = result["evaluation"]["winner"]
            category_analysis[category][winner] += 1
            category_analysis[category]["total"] += 1

        return {
            "test_info": {
                "test_name": "Cost-Controlled LLM Evaluation",
                "timestamp": datetime.now().isoformat(),
                "target_samples": target_samples,
                "completed_samples": completed_samples,
                "completion_rate": (
                    round(completed_samples / target_samples, 3) if target_samples > 0 else 0
                ),
                "total_time": round(total_time, 2),
            },
            "providers": {
                "provider_a": {
                    "name": self.provider_a.provider_name,
                    "model": self.provider_a.model_name,
                    "requests": self.provider_a.request_count,
                    "tokens": self.provider_a.total_tokens,
                    "cost": round(provider_a_cost, 6),
                },
                "provider_b": {
                    "name": self.provider_b.provider_name,
                    "model": self.provider_b.model_name,
                    "requests": self.provider_b.request_count,
                    "tokens": self.provider_b.total_tokens,
                    "cost": round(provider_b_cost, 6),
                },
                "judge": {
                    "name": self.judge.judge_provider.provider_name,
                    "model": self.judge.judge_provider.model_name,
                    "evaluations": self.judge.evaluation_count,
                    "tokens": self.judge.judge_provider.total_tokens,
                    "cost": round(judge_cost, 6),
                },
            },
            "results": {
                "provider_a_wins": provider_a_wins,
                "provider_b_wins": provider_b_wins,
                "ties": ties,
                "win_rate_a": (
                    round(provider_a_wins / completed_samples, 3) if completed_samples > 0 else 0
                ),
                "win_rate_b": (
                    round(provider_b_wins / completed_samples, 3) if completed_samples > 0 else 0
                ),
            },
            "cost_analysis": {
                "total_cost": round(total_cost, 6),
                "budget_used": round((total_cost / self.max_cost_usd) * 100, 1),
                "avg_cost_per_sample": round(avg_cost_per_sample, 6),
                "cost_breakdown": {
                    "provider_a": (
                        round((provider_a_cost / total_cost) * 100, 1) if total_cost > 0 else 0
                    ),
                    "provider_b": (
                        round((provider_b_cost / total_cost) * 100, 1) if total_cost > 0 else 0
                    ),
                    "judge": round((judge_cost / total_cost) * 100, 1) if total_cost > 0 else 0,
                },
            },
            "token_analysis": {
                "total_tokens": total_tokens,
                "avg_tokens_per_sample": (
                    round(total_tokens / completed_samples, 1) if completed_samples > 0 else 0
                ),
                "token_distribution": {
                    "provider_a": self.provider_a.total_tokens,
                    "provider_b": self.provider_b.total_tokens,
                    "judge": self.judge.judge_provider.total_tokens,
                },
            },
            "performance": {
                "avg_response_time": round(avg_response_time, 3),
                "avg_evaluation_time": round(avg_eval_time, 3),
                "throughput": round(completed_samples / total_time, 2),
                "total_time": round(total_time, 2),
            },
            "category_analysis": category_analysis,
            "sample_results": self.test_results[:5],  # 只保存前5个详细结果
        }

    def _save_cost_results(self, summary: Dict[str, Any]):
        """保存成本控制测试结果"""
        results_dir = Path("logs/cost_controlled_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"cost_controlled_test_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n📁 详细结果已保存到: {results_file}")

    def display_cost_summary(self, summary: Dict[str, Any]):
        """显示成本控制测试汇总"""
        if not summary:
            return

        logger.info("\n" + "=" * 70)
        logger.info("📊 成本控制 LLM 评估测试汇总")
        logger.info("=" * 70)

        test_info = summary["test_info"]
        providers = summary["providers"]
        results = summary["results"]
        cost = summary["cost_analysis"]
        tokens = summary["token_analysis"]
        performance = summary["performance"]

        # 基本信息
        logger.info(f"🕒 测试时间: {test_info['timestamp']}")
        logger.info(f"📊 目标样本: {test_info['target_samples']}")
        logger.info(
            f"✅ 完成样本: {test_info['completed_samples']} ({test_info['completion_rate']:.1%})"
        )
        logger.info(f"⏱️ 总耗时: {test_info['total_time']}秒")
        logger.info(f"🚀 吞吐量: {performance['throughput']} 样本/秒")

        # 提供商对比
        logger.info(f"\n🤖 提供商对比:")
        logger.info(
            f"  Provider A: {providers['provider_a']['name']} ({providers['provider_a']['model']})"
        )
        logger.info(
            f"    请求数: {providers['provider_a']['requests']} | Tokens: {providers['provider_a']['tokens']:,} | 成本: ${providers['provider_a']['cost']}"
        )
        logger.info(
            f"  Provider B: {providers['provider_b']['name']} ({providers['provider_b']['model']})"
        )
        logger.info(
            f"    请求数: {providers['provider_b']['requests']} | Tokens: {providers['provider_b']['tokens']:,} | 成本: ${providers['provider_b']['cost']}"
        )
        logger.info(f"  Judge: {providers['judge']['name']} ({providers['judge']['model']})")
        logger.info(
            f"    评估数: {providers['judge']['evaluations']} | Tokens: {providers['judge']['tokens']:,} | 成本: ${providers['judge']['cost']}"
        )

        # 比赛结果
        logger.info(f"\n🏆 比赛结果:")
        logger.info(
            f"  Provider A 获胜: {results['provider_a_wins']} 次 ({results['win_rate_a']:.1%})"
        )
        logger.info(
            f"  Provider B 获胜: {results['provider_b_wins']} 次 ({results['win_rate_b']:.1%})"
        )
        logger.info(f"  平局: {results['ties']} 次")

        # 成本分析
        logger.info(f"\n💰 成本分析:")
        logger.info(f"  总成本: ${cost['total_cost']}")
        logger.info(f"  预算使用: {cost['budget_used']}%")
        logger.info(f"  每样本平均成本: ${cost['avg_cost_per_sample']}")
        logger.info(
            f"  成本分布: A-{cost['cost_breakdown']['provider_a']}% | B-{cost['cost_breakdown']['provider_b']}% | Judge-{cost['cost_breakdown']['judge']}%"
        )

        # Token分析
        logger.info(f"\n🔤 Token分析:")
        logger.info(f"  总Token数: {tokens['total_tokens']:,}")
        logger.info(f"  每样本平均Token: {tokens['avg_tokens_per_sample']:.0f}")

        # 结论
        logger.info(f"\n🎯 测试结论:")
        if results["provider_a_wins"] > results["provider_b_wins"]:
            winner = providers["provider_a"]["name"]
            winner_rate = results["win_rate_a"]
        elif results["provider_b_wins"] > results["provider_a_wins"]:
            winner = providers["provider_b"]["name"]
            winner_rate = results["win_rate_b"]
        else:
            winner = "平局"
            winner_rate = 0.5

        if winner_rate >= 0.6:
            logger.info(f"  🎉 {winner} 表现更优秀 (胜率: {winner_rate:.1%})")
        else:
            logger.info(f"  ⚖️ 两个模型表现相近")

        logger.info(
            f"  💡 成本效率: ${cost['total_cost']:.4f} 总成本，每样本 ${cost['avg_cost_per_sample']:.6f}"
        )

        # 成本预估
        if cost["total_cost"] > 0:
            cost_per_100 = cost["avg_cost_per_sample"] * 100
            cost_per_1000 = cost["avg_cost_per_sample"] * 1000
            logger.info(
                f"  📈 成本预估: 100样本约${cost_per_100:.3f} | 1000样本约${cost_per_1000:.2f}"
            )


async def main():
    """主函数"""
    logger.info("🎯 成本控制的大规模LLM评估测试")
    logger.info("使用超便宜模型组合进行100样本评估")

    # 创建测试运行器
    tester = CostControlledTestRunner(max_cost_usd=1.0)  # 预算$1

    # 运行测试
    summary = await tester.run_cost_controlled_test(target_samples=100)

    if summary:
        # 显示结果
        tester.display_cost_summary(summary)
        logger.info("\n✅ 成本控制测试完成！")
    else:
        logger.error("❌ 测试失败")

    return summary


if __name__ == "__main__":
    asyncio.run(main())

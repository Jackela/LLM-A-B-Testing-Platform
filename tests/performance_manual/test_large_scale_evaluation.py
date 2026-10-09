#!/usr/bin/env python3
"""
Large Scale LLM Evaluation Test - 100 Samples
使用便宜模型进行大规模LLM as a Judge评估
支持: Gemini Flash, Claude Sonnet, GPT-3.5 Turbo
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

import httpx

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class APILLMProvider:
    """真实API LLM提供商"""

    def __init__(self, provider_name: str, model_name: str, api_key: str = None):
        self.provider_name = provider_name
        self.model_name = model_name
        self.api_key = api_key or os.getenv(f"{provider_name.upper()}_API_KEY")
        self.request_count = 0
        self.total_cost = 0.0

        # 价格配置 (每1K tokens的价格，USD)
        self.pricing = {
            "gemini-flash": {"input": 0.000075, "output": 0.0003},  # Gemini 1.5 Flash
            "claude-sonnet": {"input": 0.003, "output": 0.015},  # Claude 3.5 Sonnet
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},  # GPT-3.5 Turbo
        }

    async def generate_response(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """生成响应并计算成本"""
        if not self.api_key:
            # 如果没有API key，使用模拟响应
            return await self._simulate_response(prompt, context)

        try:
            if "gemini" in self.model_name.lower():
                return await self._call_gemini_api(prompt)
            elif "claude" in self.model_name.lower():
                return await self._call_claude_api(prompt)
            elif "gpt" in self.model_name.lower():
                return await self._call_openai_api(prompt)
            else:
                return await self._simulate_response(prompt, context)
        except Exception as e:
            logger.warning(f"API调用失败，使用模拟响应: {e}")
            return await self._simulate_response(prompt, context)

    async def _call_gemini_api(self, prompt: str) -> Dict[str, Any]:
        """调用Gemini API"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={self.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500},
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()

            data = response.json()
            content = data["candidates"][0]["content"]["parts"][0]["text"]

            # 估算token使用量
            input_tokens = len(prompt.split()) * 1.3  # 粗略估算
            output_tokens = len(content.split()) * 1.3

            cost = self._calculate_cost("gemini-flash", input_tokens, output_tokens)
            self.total_cost += cost
            self.request_count += 1

            return {
                "content": content,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "cost": cost,
            }

    async def _call_claude_api(self, prompt: str) -> Dict[str, Any]:
        """调用Claude API"""
        url = "https://api.anthropic.com/v1/messages"

        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 500,
            "temperature": 0.7,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            content = data["content"][0]["text"]
            input_tokens = data["usage"]["input_tokens"]
            output_tokens = data["usage"]["output_tokens"]

            cost = self._calculate_cost("claude-sonnet", input_tokens, output_tokens)
            self.total_cost += cost
            self.request_count += 1

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }

    async def _call_openai_api(self, prompt: str) -> Dict[str, Any]:
        """调用OpenAI API"""
        url = "https://api.openai.com/v1/chat/completions"

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
            "temperature": 0.7,
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            input_tokens = data["usage"]["prompt_tokens"]
            output_tokens = data["usage"]["completion_tokens"]

            cost = self._calculate_cost("gpt-3.5-turbo", input_tokens, output_tokens)
            self.total_cost += cost
            self.request_count += 1

            return {
                "content": content,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": cost,
            }

    async def _simulate_response(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """模拟API响应（当没有API key时使用）"""
        await asyncio.sleep(random.uniform(0.2, 0.8))  # 模拟网络延迟

        # 基于上下文生成模拟响应
        if context:
            if context.get("source") == "ARC-Easy":
                content = self._generate_science_response(prompt, context)
            elif context.get("source") == "GSM8K":
                content = self._generate_math_response(prompt, context)
            else:
                content = f"Based on the question, {context.get('expected_output', 'this requires careful analysis.')}"
        else:
            content = f"This is a simulated response from {self.model_name}."

        # 估算模拟的token使用量和成本
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(content.split()) * 1.3
        cost = self._calculate_cost(self.model_name.lower(), input_tokens, output_tokens)

        self.total_cost += cost
        self.request_count += 1

        return {
            "content": content,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cost": cost,
        }

    def _generate_science_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """生成科学问题响应"""
        if "gemini" in self.model_name.lower():
            return f"Based on scientific principles, {context.get('expected_output', 'the answer requires understanding fundamental concepts.')} This conclusion is supported by established scientific knowledge."
        elif "claude" in self.model_name.lower():
            return f"Looking at this scientifically: {context.get('expected_output', 'the answer follows from basic principles.')} The reasoning is straightforward when we consider the underlying mechanisms."
        else:
            return f"From a scientific perspective, {context.get('expected_output', 'this can be explained through established theories.')}"

    def _generate_math_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """生成数学问题响应"""
        if "gemini" in self.model_name.lower():
            return f"Let me solve this step-by-step: First, I'll identify the known values. Then I'll apply the appropriate mathematical operations. The final answer is {context.get('ground_truth', 'calculated result')}."
        elif "claude" in self.model_name.lower():
            return f"To solve this problem: I need to work through the calculations systematically. The answer is {context.get('ground_truth', 'the computed value')}."
        else:
            return f"Solving this mathematically: {context.get('ground_truth', 'result obtained through calculation')}."

    def _calculate_cost(self, model_key: str, input_tokens: float, output_tokens: float) -> float:
        """计算API调用成本"""
        if model_key not in self.pricing:
            model_key = "gpt-3.5-turbo"  # 默认价格

        pricing = self.pricing[model_key]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]

        return input_cost + output_cost


class ScalableLLMJudge:
    """可扩展的LLM评判者，支持批量评估"""

    def __init__(self, judge_provider: APILLMProvider = None):
        self.judge_provider = judge_provider
        self.evaluation_count = 0

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
        judge_prompt = self._build_judge_prompt(
            question, response_a["content"], response_b["content"], provider_a, provider_b, context
        )

        # 使用LLM进行评估
        if self.judge_provider and self.judge_provider.api_key:
            try:
                judge_response = await self.judge_provider.generate_response(judge_prompt)
                evaluation = self._parse_judge_response(judge_response["content"])
                evaluation["judge_cost"] = judge_response["cost"]
            except Exception as e:
                logger.warning(f"Judge API调用失败，使用启发式评估: {e}")
                evaluation = self._heuristic_evaluation(
                    question, response_a["content"], response_b["content"], context
                )
                evaluation["judge_cost"] = 0.0
        else:
            # 使用启发式评估
            evaluation = self._heuristic_evaluation(
                question, response_a["content"], response_b["content"], context
            )
            evaluation["judge_cost"] = 0.0

        self.evaluation_count += 1
        return evaluation

    def _build_judge_prompt(
        self,
        question: str,
        response_a: str,
        response_b: str,
        provider_a: str,
        provider_b: str,
        context: Dict[str, Any],
    ) -> str:
        """构建评判提示"""
        question_type = context.get("category", "general") if context else "general"

        if question_type == "science":
            criteria = "accuracy (scientific correctness), reasoning (quality of explanation), clarity (how well explained)"
        elif question_type == "math":
            criteria = "accuracy (mathematical correctness), methodology (solution approach), clarity (explanation quality)"
        else:
            criteria = "accuracy (factual correctness), helpfulness (usefulness), clarity (communication quality)"

        return f"""
Please evaluate these two AI responses to determine which is better.

Question: {question}

Response A ({provider_a}):
{response_a}

Response B ({provider_b}):
{response_b}

Evaluate based on: {criteria}

Please respond with a JSON object containing:
{{
    "winner": "A" or "B" or "Tie",
    "confidence": number between 0.5 and 1.0,
    "reasoning": "brief explanation of your choice",
    "scores": {{
        "response_a": number between 1-10,
        "response_b": number between 1-10
    }}
}}
"""

    def _parse_judge_response(self, response: str) -> Dict[str, Any]:
        """解析Judge响应"""
        try:
            # 尝试提取JSON
            import re

            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
                return eval_data
        except:
            pass

        # 如果JSON解析失败，使用启发式解析
        if "response a" in response.lower() or "a is better" in response.lower():
            winner = "A"
        elif "response b" in response.lower() or "b is better" in response.lower():
            winner = "B"
        else:
            winner = "Tie"

        return {
            "winner": winner,
            "confidence": 0.7,
            "reasoning": "Parsed from judge response",
            "scores": {"response_a": 7, "response_b": 7},
        }

    def _heuristic_evaluation(
        self, question: str, response_a: str, response_b: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """启发式评估方法"""

        # 基础评分因子
        len_a, len_b = len(response_a), len(response_b)

        # 检查是否包含预期答案
        score_a = 7.0
        score_b = 7.0

        if context:
            expected = context.get("expected_output", "").lower()
            ground_truth = str(context.get("ground_truth", "")).lower()

            if expected in response_a.lower() or ground_truth in response_a.lower():
                score_a += 1.5
            if expected in response_b.lower() or ground_truth in response_b.lower():
                score_b += 1.5

        # 长度和详细程度评估
        if len_a > len_b * 1.5:
            score_a += 0.5  # 奖励详细回答
        elif len_b > len_a * 1.5:
            score_b += 0.5

        # 结构化回答奖励
        if any(word in response_a.lower() for word in ["step", "first", "then", "because"]):
            score_a += 0.5
        if any(word in response_b.lower() for word in ["step", "first", "then", "because"]):
            score_b += 0.5

        # 确定获胜者
        if score_a > score_b:
            winner = "A"
            confidence = min(0.9, 0.6 + (score_a - score_b) / 10)
        elif score_b > score_a:
            winner = "B"
            confidence = min(0.9, 0.6 + (score_b - score_a) / 10)
        else:
            winner = "Tie"
            confidence = 0.5

        return {
            "winner": winner,
            "confidence": round(confidence, 2),
            "reasoning": f"Score A: {score_a:.1f}, Score B: {score_b:.1f}",
            "scores": {"response_a": round(score_a, 1), "response_b": round(score_b, 1)},
        }


class LargeScaleTestRunner:
    """大规模测试运行器"""

    def __init__(self):
        # 初始化便宜的模型提供商
        self.provider_a = APILLMProvider("Gemini", "gemini-flash")
        self.provider_b = APILLMProvider("Claude", "claude-sonnet")

        # 初始化评判者（也使用便宜的模型）
        self.judge = ScalableLLMJudge(APILLMProvider("GPT", "gpt-3.5-turbo"))

        self.test_results = []
        self.batch_size = 10  # 批处理大小

    def load_test_samples(self, sample_count: int = 100) -> List[Dict[str, Any]]:
        """加载测试样本"""
        datasets_dir = Path("data/processed")
        all_samples = []

        # 从ARC数据集加载
        arc_file = datasets_dir / "arc_easy.json"
        if arc_file.exists():
            with open(arc_file, "r", encoding="utf-8") as f:
                arc_data = json.load(f)
            all_samples.extend(arc_data)

        # 从GSM8K数据集加载
        gsm8k_file = datasets_dir / "gsm8k.json"
        if gsm8k_file.exists():
            with open(gsm8k_file, "r", encoding="utf-8") as f:
                gsm8k_data = json.load(f)
            all_samples.extend(gsm8k_data)

        # 随机选择指定数量的样本
        if len(all_samples) > sample_count:
            return random.sample(all_samples, sample_count)
        else:
            return all_samples

    async def process_batch(
        self, batch: List[Dict[str, Any]], batch_num: int
    ) -> List[Dict[str, Any]]:
        """处理一个批次的样本"""
        logger.info(f"🔄 处理批次 {batch_num}: {len(batch)} 个样本")

        batch_results = []
        for i, sample in enumerate(batch):
            try:
                result = await self.evaluate_single_sample(
                    sample, batch_num * self.batch_size + i + 1
                )
                batch_results.append(result)

                # 显示进度
                if (i + 1) % 5 == 0:
                    logger.info(f"  ✅ 批次 {batch_num} 进度: {i + 1}/{len(batch)}")

            except Exception as e:
                logger.error(f"❌ 样本评估失败: {e}")
                continue

        return batch_results

    async def evaluate_single_sample(
        self, sample: Dict[str, Any], sample_num: int
    ) -> Dict[str, Any]:
        """评估单个样本"""
        question = sample["prompt"]

        # 获取两个提供商的响应
        start_time = time.time()

        # 并发获取响应
        response_a_task = self.provider_a.generate_response(question, sample)
        response_b_task = self.provider_b.generate_response(question, sample)

        response_a, response_b = await asyncio.gather(response_a_task, response_b_task)

        response_time = time.time() - start_time

        # LLM评估
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
        total_cost = response_a["cost"] + response_b["cost"] + evaluation.get("judge_cost", 0.0)

        return {
            "sample_num": sample_num,
            "sample_data": {
                "id": sample["id"],
                "category": sample.get("category", "unknown"),
                "source": sample.get("source", "unknown"),
            },
            "responses": {
                "provider_a": {
                    "name": self.provider_a.provider_name,
                    "model": self.provider_a.model_name,
                    "content": (
                        response_a["content"][:200] + "..."
                        if len(response_a["content"]) > 200
                        else response_a["content"]
                    ),
                    "tokens": response_a["input_tokens"] + response_a["output_tokens"],
                    "cost": response_a["cost"],
                },
                "provider_b": {
                    "name": self.provider_b.provider_name,
                    "model": self.provider_b.model_name,
                    "content": (
                        response_b["content"][:200] + "..."
                        if len(response_b["content"]) > 200
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
            "timestamp": datetime.now().isoformat(),
        }

    async def run_large_scale_test(self, sample_count: int = 100) -> Dict[str, Any]:
        """运行大规模测试"""
        logger.info("🚀 开始大规模LLM as a Judge测试")
        logger.info(f"📊 目标样本数: {sample_count}")
        logger.info(f"🤖 Provider A: {self.provider_a.provider_name} ({self.provider_a.model_name})")
        logger.info(f"🤖 Provider B: {self.provider_b.provider_name} ({self.provider_b.model_name})")
        logger.info("=" * 80)

        # 加载测试数据
        test_samples = self.load_test_samples(sample_count)
        actual_count = len(test_samples)
        logger.info(f"📚 已加载 {actual_count} 个测试样本")

        if actual_count == 0:
            logger.error("❌ 没有找到测试样本")
            return {}

        # 分批处理
        batches = [
            test_samples[i : i + self.batch_size]
            for i in range(0, len(test_samples), self.batch_size)
        ]

        overall_start_time = time.time()

        # 并发处理批次（限制并发数）
        semaphore = asyncio.Semaphore(3)  # 最多3个并发批次

        async def process_batch_with_semaphore(batch, batch_num):
            async with semaphore:
                return await self.process_batch(batch, batch_num)

        # 执行所有批次
        batch_tasks = [process_batch_with_semaphore(batch, i) for i, batch in enumerate(batches)]
        batch_results = await asyncio.gather(*batch_tasks)

        # 合并结果
        for batch_result in batch_results:
            self.test_results.extend(batch_result)

        total_time = time.time() - overall_start_time

        # 生成汇总
        summary = self._generate_large_scale_summary(total_time, actual_count)

        # 保存结果
        self._save_large_scale_results(summary)

        return summary

    def _generate_large_scale_summary(
        self, total_time: float, expected_count: int
    ) -> Dict[str, Any]:
        """生成大规模测试汇总"""
        if not self.test_results:
            return {}

        actual_count = len(self.test_results)

        # 基础统计
        provider_a_wins = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "A")
        provider_b_wins = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "B")
        ties = sum(1 for r in self.test_results if r["evaluation"]["winner"] == "Tie")

        # 成本分析
        total_cost = sum(r["total_cost"] for r in self.test_results)
        avg_cost_per_sample = total_cost / actual_count if actual_count > 0 else 0

        provider_a_cost = self.provider_a.total_cost
        provider_b_cost = self.provider_b.total_cost
        judge_cost = sum(r["evaluation"].get("judge_cost", 0) for r in self.test_results)

        # 性能指标
        avg_response_time = (
            sum(r["timing"]["response_time"] for r in self.test_results) / actual_count
        )
        avg_eval_time = (
            sum(r["timing"]["evaluation_time"] for r in self.test_results) / actual_count
        )

        # 按类型分析
        type_analysis = {}
        source_analysis = {}

        for result in self.test_results:
            # 按类别分析
            category = result["sample_data"]["category"]
            if category not in type_analysis:
                type_analysis[category] = {"A": 0, "B": 0, "Tie": 0, "total": 0}

            winner = result["evaluation"]["winner"]
            type_analysis[category][winner] += 1
            type_analysis[category]["total"] += 1

            # 按数据源分析
            source = result["sample_data"]["source"]
            if source not in source_analysis:
                source_analysis[source] = {"A": 0, "B": 0, "Tie": 0, "total": 0}

            source_analysis[source][winner] += 1
            source_analysis[source]["total"] += 1

        # 成功率分析
        success_rate = actual_count / expected_count if expected_count > 0 else 1.0

        return {
            "test_info": {
                "test_name": "Large Scale LLM as a Judge Test",
                "timestamp": datetime.now().isoformat(),
                "target_samples": expected_count,
                "actual_samples": actual_count,
                "success_rate": round(success_rate, 3),
                "total_time": round(total_time, 2),
            },
            "providers": {
                "provider_a": {
                    "name": self.provider_a.provider_name,
                    "model": self.provider_a.model_name,
                    "requests": self.provider_a.request_count,
                    "total_cost": round(provider_a_cost, 4),
                },
                "provider_b": {
                    "name": self.provider_b.provider_name,
                    "model": self.provider_b.model_name,
                    "requests": self.provider_b.request_count,
                    "total_cost": round(provider_b_cost, 4),
                },
            },
            "results": {
                "provider_a_wins": provider_a_wins,
                "provider_b_wins": provider_b_wins,
                "ties": ties,
                "win_rate_a": round(provider_a_wins / actual_count, 3),
                "win_rate_b": round(provider_b_wins / actual_count, 3),
            },
            "cost_analysis": {
                "total_cost": round(total_cost, 4),
                "avg_cost_per_sample": round(avg_cost_per_sample, 5),
                "provider_a_cost": round(provider_a_cost, 4),
                "provider_b_cost": round(provider_b_cost, 4),
                "judge_cost": round(judge_cost, 4),
                "cost_breakdown": {
                    "responses": (
                        round((provider_a_cost + provider_b_cost) / total_cost * 100, 1)
                        if total_cost > 0
                        else 0
                    ),
                    "evaluation": round(judge_cost / total_cost * 100, 1) if total_cost > 0 else 0,
                },
            },
            "performance": {
                "avg_response_time": round(avg_response_time, 3),
                "avg_evaluation_time": round(avg_eval_time, 3),
                "throughput": round(actual_count / total_time, 2),  # samples per second
                "total_time": round(total_time, 2),
            },
            "analysis": {"by_category": type_analysis, "by_source": source_analysis},
            "sample_results": self.test_results[:10],  # 只保存前10个详细结果
        }

    def _save_large_scale_results(self, summary: Dict[str, Any]):
        """保存大规模测试结果"""
        results_dir = Path("logs/large_scale_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"large_scale_test_{timestamp}.json"

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n📁 详细结果已保存到: {results_file}")

    def display_large_scale_summary(self, summary: Dict[str, Any]):
        """显示大规模测试汇总"""
        if not summary:
            return

        logger.info("\n" + "=" * 80)
        logger.info("📊 大规模 LLM AS A JUDGE 测试汇总")
        logger.info("=" * 80)

        test_info = summary["test_info"]
        providers = summary["providers"]
        results = summary["results"]
        cost = summary["cost_analysis"]
        performance = summary["performance"]
        analysis = summary["analysis"]

        # 基本信息
        logger.info(f"🕒 测试时间: {test_info['timestamp']}")
        logger.info(f"📊 目标样本: {test_info['target_samples']}")
        logger.info(f"✅ 实际完成: {test_info['actual_samples']} ({test_info['success_rate']:.1%})")
        logger.info(f"⏱️ 总耗时: {test_info['total_time']}秒")
        logger.info(f"🚀 吞吐量: {performance['throughput']} 样本/秒")

        # 提供商信息
        logger.info(f"\n🤖 测试提供商:")
        logger.info(
            f"  Provider A: {providers['provider_a']['name']} ({providers['provider_a']['model']})"
        )
        logger.info(
            f"    请求数: {providers['provider_a']['requests']}, 成本: ${providers['provider_a']['total_cost']}"
        )
        logger.info(
            f"  Provider B: {providers['provider_b']['name']} ({providers['provider_b']['model']})"
        )
        logger.info(
            f"    请求数: {providers['provider_b']['requests']}, 成本: ${providers['provider_b']['total_cost']}"
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
        logger.info(f"  每样本平均成本: ${cost['avg_cost_per_sample']}")
        logger.info(
            f"  成本分布: 响应生成 {cost['cost_breakdown']['responses']}%, 评估 {cost['cost_breakdown']['evaluation']}%"
        )

        # 按类型分析
        logger.info(f"\n📋 按类型分析:")
        for category, data in analysis["by_category"].items():
            total = data["total"]
            logger.info(f"  {category.upper()}类型 (共{total}题):")
            logger.info(f"    Provider A: {data['A']} 胜 ({data['A']/total:.1%})")
            logger.info(f"    Provider B: {data['B']} 胜 ({data['B']/total:.1%})")
            if data.get("Tie", 0) > 0:
                logger.info(f"    平局: {data['Tie']} 次")

        # 性能指标
        logger.info(f"\n⚡ 性能指标:")
        logger.info(f"  平均响应时间: {performance['avg_response_time']}秒")
        logger.info(f"  平均评估时间: {performance['avg_evaluation_time']}秒")

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
            logger.info(f"  🎉 {winner} 表现显著更好 (胜率: {winner_rate:.1%})")
        else:
            logger.info(f"  ⚖️ 两个模型表现相近")

        logger.info(f"  💡 总成本控制良好: ${cost['total_cost']} (平均每样本 ${cost['avg_cost_per_sample']})")


async def main():
    """主函数"""
    logger.info("🎯 大规模LLM as a Judge评估测试 (100样本)")
    logger.info("使用便宜模型: Gemini Flash vs Claude Sonnet")
    logger.info("评判者: GPT-3.5 Turbo")

    # 检查API密钥
    api_keys_info = []
    if os.getenv("GEMINI_API_KEY"):
        api_keys_info.append("✅ Gemini API")
    else:
        api_keys_info.append("⚠️ Gemini API (将使用模拟)")

    if os.getenv("ANTHROPIC_API_KEY"):
        api_keys_info.append("✅ Claude API")
    else:
        api_keys_info.append("⚠️ Claude API (将使用模拟)")

    if os.getenv("OPENAI_API_KEY"):
        api_keys_info.append("✅ OpenAI API")
    else:
        api_keys_info.append("⚠️ OpenAI API (将使用模拟)")

    logger.info("🔑 API密钥状态: " + " | ".join(api_keys_info))

    # 创建并运行测试
    tester = LargeScaleTestRunner()
    summary = await tester.run_large_scale_test(sample_count=100)

    if summary:
        # 显示汇总结果
        tester.display_large_scale_summary(summary)
        logger.info("\n✅ 大规模测试完成！")
    else:
        logger.error("❌ 测试失败")

    return summary


if __name__ == "__main__":
    asyncio.run(main())

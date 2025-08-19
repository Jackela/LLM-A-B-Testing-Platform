#!/usr/bin/env python3
"""
Real API Integration Test
真实API集成测试 - 使用实际的OpenAI, Anthropic, Google API
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API配置信息"""

    provider: str
    model: str
    api_key: str
    endpoint: str
    pricing: Dict[str, float]


class RealAPIProvider:
    """真实API提供商 - 支持OpenAI, Anthropic, Google API"""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0

        # 加载API配置
        self.api_configs = self._load_api_configs()

    def _load_api_configs(self) -> Dict[str, APIConfig]:
        """加载API配置"""
        configs = {}

        # OpenAI配置
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            configs["openai"] = APIConfig(
                provider="openai",
                model="gpt-4o-mini",
                api_key=openai_key,
                endpoint="https://api.openai.com/v1/chat/completions",
                pricing={"input": 0.00015, "output": 0.0006},  # per 1K tokens
            )

        # Anthropic配置
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            configs["anthropic"] = APIConfig(
                provider="anthropic",
                model="claude-3-haiku-20240307",
                api_key=anthropic_key,
                endpoint="https://api.anthropic.com/v1/messages",
                pricing={"input": 0.00025, "output": 0.00125},  # per 1K tokens
            )

        # Google AI配置
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_key:
            configs["google"] = APIConfig(
                provider="google",
                model="gemini-1.5-flash",
                api_key=google_key,
                endpoint="https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent",
                pricing={"input": 0.000075, "output": 0.0003},  # per 1K tokens
            )

        return configs

    async def __aenter__(self):
        """异步上下文管理器入口"""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        if self.session:
            await self.session.close()

    async def call_openai_api(self, prompt: str, config: APIConfig) -> Dict[str, Any]:
        """调用OpenAI API"""
        headers = {"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"}

        payload = {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 200,
            "temperature": 0.7,
        }

        async with self.session.post(config.endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()

                # 提取响应信息
                message = data["choices"][0]["message"]["content"]
                usage = data["usage"]
                input_tokens = usage["prompt_tokens"]
                output_tokens = usage["completion_tokens"]
                total_tokens = usage["total_tokens"]

                # 计算成本
                cost = (
                    input_tokens / 1000 * config.pricing["input"]
                    + output_tokens / 1000 * config.pricing["output"]
                )

                return {
                    "success": True,
                    "content": message,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                    "provider": config.provider,
                    "model": config.model,
                }
            else:
                error_text = await response.text()
                logger.error(f"OpenAI API error {response.status}: {error_text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text}",
                    "provider": config.provider,
                    "model": config.model,
                }

    async def call_anthropic_api(self, prompt: str, config: APIConfig) -> Dict[str, Any]:
        """调用Anthropic API"""
        headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }

        payload = {
            "model": config.model,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }

        async with self.session.post(config.endpoint, headers=headers, json=payload) as response:
            if response.status == 200:
                data = await response.json()

                # 提取响应信息
                message = data["content"][0]["text"]
                usage = data["usage"]
                input_tokens = usage["input_tokens"]
                output_tokens = usage["output_tokens"]
                total_tokens = input_tokens + output_tokens

                # 计算成本
                cost = (
                    input_tokens / 1000 * config.pricing["input"]
                    + output_tokens / 1000 * config.pricing["output"]
                )

                return {
                    "success": True,
                    "content": message,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "cost": cost,
                    "provider": config.provider,
                    "model": config.model,
                }
            else:
                error_text = await response.text()
                logger.error(f"Anthropic API error {response.status}: {error_text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text}",
                    "provider": config.provider,
                    "model": config.model,
                }

    async def call_google_api(self, prompt: str, config: APIConfig) -> Dict[str, Any]:
        """调用Google Gemini API"""
        url = f"{config.endpoint}?key={config.api_key}"

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"maxOutputTokens": 200, "temperature": 0.7},
        }

        async with self.session.post(url, json=payload) as response:
            if response.status == 200:
                data = await response.json()

                # 提取响应信息
                if "candidates" in data and data["candidates"]:
                    candidate = data["candidates"][0]
                    message = candidate["content"]["parts"][0]["text"]

                    # Gemini API通常不返回详细的token使用信息，我们需要估算
                    # 这里使用简单的估算方法
                    input_tokens = len(prompt.split()) * 1.3  # 估算input tokens
                    output_tokens = len(message.split()) * 1.3  # 估算output tokens
                    total_tokens = int(input_tokens + output_tokens)

                    # 计算成本
                    cost = (
                        input_tokens / 1000 * config.pricing["input"]
                        + output_tokens / 1000 * config.pricing["output"]
                    )

                    return {
                        "success": True,
                        "content": message,
                        "input_tokens": int(input_tokens),
                        "output_tokens": int(output_tokens),
                        "total_tokens": total_tokens,
                        "cost": cost,
                        "provider": config.provider,
                        "model": config.model,
                    }
                else:
                    error_msg = "No valid response from Google API"
                    logger.error(error_msg)
                    return {
                        "success": False,
                        "error": error_msg,
                        "provider": config.provider,
                        "model": config.model,
                    }
            else:
                error_text = await response.text()
                logger.error(f"Google API error {response.status}: {error_text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status}: {error_text}",
                    "provider": config.provider,
                    "model": config.model,
                }

    async def generate_response(self, prompt: str, provider: str) -> Dict[str, Any]:
        """生成响应 - 路由到对应的API"""
        if provider not in self.api_configs:
            return {
                "success": False,
                "error": f"Provider {provider} not configured. Please set API key.",
                "provider": provider,
                "model": "unknown",
            }

        config = self.api_configs[provider]

        try:
            if provider == "openai":
                result = await self.call_openai_api(prompt, config)
            elif provider == "anthropic":
                result = await self.call_anthropic_api(prompt, config)
            elif provider == "google":
                result = await self.call_google_api(prompt, config)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported provider: {provider}",
                    "provider": provider,
                    "model": "unknown",
                }

            if result["success"]:
                self.request_count += 1
                self.total_tokens += result["total_tokens"]
                self.total_cost += result["cost"]

            return result

        except Exception as e:
            logger.error(f"Error calling {provider} API: {str(e)}")
            return {"success": False, "error": str(e), "provider": provider, "model": config.model}


class RealAPITestRunner:
    """真实API测试运行器"""

    def __init__(self):
        self.results = []

    async def run_simple_test(self, sample_size: int = 5) -> Dict[str, Any]:
        """运行简单的真实API测试"""
        logger.info(f"开始运行真实API测试 - {sample_size}个样本")

        start_time = time.time()

        # 测试提示
        test_prompts = [
            "What is the capital of France?",
            "Explain photosynthesis in simple terms.",
            "What is 25 × 17?",
            "Name three benefits of exercise.",
            "What causes seasons on Earth?",
        ]

        results = {
            "test_info": {
                "test_name": "Real API Integration Test",
                "timestamp": datetime.now().isoformat(),
                "sample_size": sample_size,
                "test_prompts": test_prompts[:sample_size],
            },
            "provider_results": {},
            "summary": {},
        }

        async with RealAPIProvider() as provider:
            # 检查可用的API配置
            available_providers = list(provider.api_configs.keys())
            logger.info(f"可用的API提供商: {available_providers}")

            if not available_providers:
                logger.error("没有配置任何API密钥！请设置以下环境变量:")
                logger.error("- OPENAI_API_KEY (OpenAI)")
                logger.error("- ANTHROPIC_API_KEY (Anthropic)")
                logger.error("- GOOGLE_API_KEY 或 GEMINI_API_KEY (Google)")
                return {
                    "error": "No API keys configured",
                    "instructions": "Please set API keys in environment variables",
                }

            # 为每个提供商测试API调用
            for provider_name in available_providers:
                logger.info(f"测试 {provider_name} API...")
                provider_results = []

                for i, prompt in enumerate(test_prompts[:sample_size]):
                    logger.info(f"  测试样本 {i+1}/{sample_size}")

                    # 调用API
                    result = await provider.generate_response(prompt, provider_name)

                    test_result = {
                        "sample_num": i + 1,
                        "prompt": prompt,
                        "result": result,
                        "timestamp": datetime.now().isoformat(),
                    }

                    provider_results.append(test_result)

                    if result["success"]:
                        logger.info(
                            f"    ✅ 成功 - {result['total_tokens']} tokens, ${result['cost']:.6f}"
                        )
                    else:
                        logger.error(f"    ❌ 失败 - {result['error']}")

                    # 添加延迟以避免速率限制
                    await asyncio.sleep(1.0)

                results["provider_results"][provider_name] = {
                    "config": {
                        "provider": provider_name,
                        "model": provider.api_configs[provider_name].model,
                        "pricing": provider.api_configs[provider_name].pricing,
                    },
                    "results": provider_results,
                    "stats": {
                        "total_requests": len(
                            [r for r in provider_results if r["result"]["success"]]
                        ),
                        "failed_requests": len(
                            [r for r in provider_results if not r["result"]["success"]]
                        ),
                        "total_tokens": sum(
                            r["result"].get("total_tokens", 0)
                            for r in provider_results
                            if r["result"]["success"]
                        ),
                        "total_cost": sum(
                            r["result"].get("cost", 0)
                            for r in provider_results
                            if r["result"]["success"]
                        ),
                    },
                }

        end_time = time.time()
        total_time = end_time - start_time

        # 计算总体统计
        total_requests = sum(
            r["stats"]["total_requests"] for r in results["provider_results"].values()
        )
        total_failures = sum(
            r["stats"]["failed_requests"] for r in results["provider_results"].values()
        )
        total_tokens = sum(r["stats"]["total_tokens"] for r in results["provider_results"].values())
        total_cost = sum(r["stats"]["total_cost"] for r in results["provider_results"].values())

        results["summary"] = {
            "total_time": total_time,
            "available_providers": len(available_providers),
            "total_requests": total_requests,
            "total_failures": total_failures,
            "success_rate": (
                total_requests / (total_requests + total_failures)
                if (total_requests + total_failures) > 0
                else 0
            ),
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
        }

        return results

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """保存测试结果"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_api_test_results_{timestamp}.json"

        # 创建结果目录
        results_dir = Path("logs/real_api_results")
        results_dir.mkdir(parents=True, exist_ok=True)

        filepath = results_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"测试结果已保存到: {filepath}")
        return filepath


async def main():
    """主函数"""
    print("🚀 真实API集成测试")
    print("=" * 50)

    # 检查环境变量
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    }

    print("📋 API密钥状态:")
    for provider, key in api_keys.items():
        status = "✅ 已配置" if key else "❌ 未配置"
        print(f"  {provider}: {status}")

    if not any(api_keys.values()):
        print("\n❌ 错误: 没有配置任何API密钥！")
        print("请设置以下环境变量之一:")
        print("  export OPENAI_API_KEY='your_openai_key'")
        print("  export ANTHROPIC_API_KEY='your_anthropic_key'")
        print("  export GOOGLE_API_KEY='your_google_key'")
        return

    print("\n🔬 开始API测试...")

    # 运行测试
    runner = RealAPITestRunner()
    results = await runner.run_simple_test(sample_size=3)

    if "error" in results:
        print(f"\n❌ 测试失败: {results['error']}")
        return

    # 保存结果
    filepath = runner.save_results(results)

    # 显示结果摘要
    summary = results["summary"]
    print(f"\n📊 测试摘要:")
    print(f"  总耗时: {summary['total_time']:.2f}秒")
    print(f"  可用提供商: {summary['available_providers']}个")
    print(f"  成功请求: {summary['total_requests']}")
    print(f"  失败请求: {summary['total_failures']}")
    print(f"  成功率: {summary['success_rate']:.1%}")
    print(f"  总Token数: {summary['total_tokens']}")
    print(f"  总成本: ${summary['total_cost']:.6f}")
    print(f"  平均成本/请求: ${summary['avg_cost_per_request']:.6f}")

    print(f"\n✅ 测试完成！结果已保存到: {filepath}")

    # 显示详细结果
    print("\n📋 详细结果:")
    for provider_name, provider_data in results["provider_results"].items():
        stats = provider_data["stats"]
        config = provider_data["config"]
        print(f"\n  {provider_name.upper()} ({config['model']}):")
        print(
            f"    成功: {stats['total_requests']}/{stats['total_requests'] + stats['failed_requests']}"
        )
        print(f"    Token数: {stats['total_tokens']}")
        print(f"    成本: ${stats['total_cost']:.6f}")


if __name__ == "__main__":
    asyncio.run(main())

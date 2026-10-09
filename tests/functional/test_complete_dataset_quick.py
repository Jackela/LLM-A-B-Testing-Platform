#!/usr/bin/env python3
"""
Quick Complete Dataset LLM Evaluation Test
快速完整数据集测试 - 500个样本演示
"""

import asyncio
import os
import sys

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging

from test_complete_dataset_evaluation import CompleteDatasetTestRunner

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def main():
    """快速演示主函数"""
    logger.info("🎯 快速完整数据集LLM评估测试 (500样本)")
    logger.info("展示完整数据集测试功能和分析能力")

    # 创建测试运行器
    tester = CompleteDatasetTestRunner(max_cost_usd=2.0)  # 降低预算

    # 运行快速测试
    summary = await tester.run_complete_dataset_test(
        test_size=500, sampling_strategy="stratified"  # 减少样本数
    )

    if summary:
        # 显示结果
        tester.display_comprehensive_summary(summary)
        logger.info("\n✅ 快速完整数据集测试完成！")

        # 显示关键统计
        logger.info("\n" + "=" * 60)
        logger.info("🎉 快速测试关键成果:")

        test_info = summary["test_info"]
        results = summary["results"]
        cost = summary["cost_analysis"]

        logger.info(
            f"📊 完成样本: {test_info['completed_samples']:,} ({test_info['completion_rate']:.1%})"
        )
        logger.info(
            f"🏆 胜率: A={results['win_rate_a']:.1%} | B={results['win_rate_b']:.1%} | 平局={results['tie_rate']:.1%}"
        )
        logger.info(f"💰 总成本: ${cost['total_cost']} (预算使用: {cost['budget_used_percentage']}%)")
        logger.info(f"⚡ 吞吐量: {summary['performance_metrics']['throughput']:.1f} 样本/秒")

        # 置信度分析
        if "confidence_analysis" in summary:
            conf = summary["confidence_analysis"]
            logger.info(f"🎯 平均置信度: {conf['mean']:.3f} (高置信度比例: {conf['high_confidence_rate']:.1%})")

        logger.info("=" * 60)

    else:
        logger.error("❌ 测试失败")

    return summary


if __name__ == "__main__":
    asyncio.run(main())

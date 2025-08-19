#!/usr/bin/env python3
"""
Merge ARC-Easy Complete Dataset Test Results
拼接ARC-Easy完整数据集测试结果
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_all_test_results() -> List[Dict[str, Any]]:
    """加载所有测试结果文件"""
    results_dir = Path("logs/complete_arc_easy_results")
    all_results = []
    
    # 找到所有结果文件，按时间排序
    result_files = []
    
    # 原始测试文件 (904样本)
    original_files = [
        "arc_easy_complete_test_interim_904_20250814_220436.json"
    ]
    
    # 最终测试文件 (1422样本)
    final_files = [
        "arc_easy_complete_dataset_1422samples_20250815_070804.json"
    ]
    
    for filename in original_files + final_files:
        file_path = results_dir / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.append({
                    'filename': filename,
                    'data': data,
                    'sample_count': len(data['test_results'])
                })
                logger.info(f"加载文件: {filename} - {len(data['test_results'])}样本")
    
    return all_results

def merge_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """合并所有测试结果"""
    
    if not all_results:
        raise ValueError("没有找到测试结果文件")
    
    # 使用最后一个文件作为基础模板
    base_result = all_results[-1]['data'].copy()
    
    # 合并所有test_results
    merged_test_results = []
    sample_id_map = set()
    
    # 先添加原始904样本的结果
    for result_file in all_results:
        for test_result in result_file['data']['test_results']:
            sample_id = test_result['sample_data']['id']
            if sample_id not in sample_id_map:
                merged_test_results.append(test_result)
                sample_id_map.add(sample_id)
    
    # 更新合并后的结果
    base_result['test_results'] = merged_test_results
    
    # 重新计算统计信息
    total_samples = 5197  # ARC-Easy完整数据集大小
    completed_tests = len(merged_test_results)
    
    # 计算成本
    total_cost = 0.0
    model_stats = {'openai': {'wins': 0, 'requests': 0, 'tokens': 0, 'cost': 0.0, 'response_times': [], 'error_count': 0},
                   'anthropic': {'wins': 0, 'requests': 0, 'tokens': 0, 'cost': 0.0, 'response_times': [], 'error_count': 0},
                   'google': {'wins': 0, 'requests': 0, 'tokens': 0, 'cost': 0.0, 'response_times': [], 'error_count': 0}}
    ties = 0
    
    for test_result in merged_test_results:
        # 计算成本
        model_a_cost = test_result['responses']['model_a']['cost']
        model_b_cost = test_result['responses']['model_b']['cost']
        judge_cost = test_result['evaluation']['judge_cost']
        total_cost += model_a_cost + model_b_cost + judge_cost
        
        # 更新模型统计
        provider_a = test_result['responses']['model_a']['provider']
        provider_b = test_result['responses']['model_b']['provider']
        
        model_stats[provider_a]['requests'] += 1
        model_stats[provider_a]['tokens'] += test_result['responses']['model_a']['tokens']
        model_stats[provider_a]['cost'] += model_a_cost
        model_stats[provider_a]['response_times'].append(test_result['responses']['model_a']['response_time'])
        
        model_stats[provider_b]['requests'] += 1
        model_stats[provider_b]['tokens'] += test_result['responses']['model_b']['tokens']
        model_stats[provider_b]['cost'] += model_b_cost
        model_stats[provider_b]['response_times'].append(test_result['responses']['model_b']['response_time'])
        
        # 更新胜负统计
        winner = test_result['evaluation']['winner']
        if winner == 'model_a':
            model_stats[provider_a]['wins'] += 1
        elif winner == 'model_b':
            model_stats[provider_b]['wins'] += 1
        else:
            ties += 1
    
    # 计算平均响应时间
    for provider, stats in model_stats.items():
        if stats['response_times']:
            stats['avg_response_time'] = sum(stats['response_times']) / len(stats['response_times'])
            del stats['response_times']
    
    # 更新测试信息
    base_result['test_info'].update({
        'test_name': 'Complete ARC-Easy Dataset Test - Merged Results',
        'merge_timestamp': datetime.now().isoformat(),
        'total_samples': total_samples,
        'completed_samples': completed_tests,
        'completion_rate': completed_tests / total_samples,
        'files_merged': [r['filename'] for r in all_results],
        'merge_note': 'Merged from original 904 samples + continuation 1422 samples = 2326 total samples'
    })
    
    # 更新汇总统计
    base_result['summary'] = {
        'total_time': sum(r['data'].get('summary', {}).get('total_time', 0) for r in all_results),
        'completed_tests': completed_tests,
        'completion_rate': completed_tests / total_samples,
        'total_cost': total_cost,
        'budget_used_percentage': (total_cost / 10.0 * 100),  # 假设预算是$10
        'avg_cost_per_test': total_cost / completed_tests if completed_tests > 0 else 0,
        'model_stats': model_stats,
        'ties': ties
    }
    
    return base_result

def save_merged_results(merged_data: Dict[str, Any]) -> Path:
    """保存合并后的结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    completed = merged_data['summary']['completed_tests']
    filename = f"arc_easy_complete_merged_{completed}samples_{timestamp}.json"
    
    results_dir = Path("logs/complete_arc_easy_results")
    filepath = results_dir / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"合并结果已保存到: {filepath}")
    return filepath

def main():
    """主函数"""
    print("🔄 开始拼接ARC-Easy完整数据集测试结果")
    print("=" * 60)
    
    # 加载所有结果
    all_results = load_all_test_results()
    
    if not all_results:
        print("❌ 未找到测试结果文件")
        return
    
    print(f"✅ 找到 {len(all_results)} 个结果文件")
    
    # 合并结果
    merged_data = merge_results(all_results)
    
    # 保存合并结果
    filepath = save_merged_results(merged_data)
    
    # 显示摘要
    summary = merged_data['summary']
    test_info = merged_data['test_info']
    
    print(f"\n📊 合并结果摘要:")
    print(f"  原始数据集大小: {test_info['total_samples']:,}")
    print(f"  完成测试: {summary['completed_tests']:,}")
    print(f"  完成率: {summary['completion_rate']:.1%}")
    print(f"  总成本: ${summary['total_cost']:.6f}")
    print(f"  预算使用: {summary['budget_used_percentage']:.1f}%")
    print(f"  平均成本/测试: ${summary['avg_cost_per_test']:.6f}")
    print(f"  平局次数: {summary['ties']:,}")
    
    print(f"\n🏆 最终模型表现:")
    for model, stats in summary['model_stats'].items():
        if stats['requests'] > 0:
            win_rate = stats['wins'] / stats['requests'] * 100
            print(f"  {model.upper()}:")
            print(f"    参与测试: {stats['requests']:,}次")
            print(f"    获胜: {stats['wins']:,}次 ({win_rate:.1f}%)")
            print(f"    Token使用: {stats['tokens']:,}")
            print(f"    成本: ${stats['cost']:.6f}")
            print(f"    错误次数: {stats['error_count']:,}")
    
    print(f"\n✅ 拼接完成！")
    print(f"📁 合并结果保存到: {filepath}")

if __name__ == "__main__":
    main()
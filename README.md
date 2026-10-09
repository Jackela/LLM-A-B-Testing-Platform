# LLM A/B Testing Platform

这里保留语言模型回答比较的实验代码、统计工具和早期平台实现。项目仍在整理，页面介绍和维护由 AI 完成。

[English](README_EN.md)

## 先看什么

- [实验来源记录](experiments/evidence.json)：原始文件摘要、报告数字和运行方式。
- [2025 年 ARC-Easy 报告](ARC_EASY_COMPLETE_TEST_REPORT.md)：历史记录，报告称完成 2,088 / 5,197 个样本，约 40.2%。对应原始结果文件没有提交，尚未独立复现；报告中的“完整验证”“生产可用”和模型优劣判断不能作为当前结论。
- [统计实现](src/domain/analytics/entities/statistical_test.py)：已经包含独立、配对和 Welch t 检验等方法；当前离线检查核对三类 t 检验与 SciPy 的结果，以及样本不足、配对长度不等和非有限数的失败行为。
- [维护入口](AGENTS.md)：后续 AI 修改时需要遵守的来源和验证约定。

## 离线检查

使用 Python 3.11，在独立环境中安装 `requirements-offline.txt`：

```bash
python -m venv .venv-offline
.venv-offline/bin/python -m pip install -r requirements-offline.txt
.venv-offline/bin/python tools/verify_experiment_evidence.py
.venv-offline/bin/python -m unittest discover -s tests/offline -v
```

这些检查覆盖统计、历史记录、共享缓存格式、无效令牌处理和依赖扫描的失败行为，不启动完整平台，也不调用模型服务。完整平台仍使用 Poetry 管理依赖；原有全量测试的 80% 覆盖要求保留。

CI 分别在 Python 3.11 和 3.12 检查这些离线合同，并审计 `poetry.lock` 中对应环境的全部运行依赖。扫描器不可用、返回格式错误或扫描失败都会使检查失败。可用 `poetry check --lock` 检查锁文件，用 `make security-scan` 执行源码和依赖检查。

## 实验入口

| 文件 | 用途 | 运行方式 |
|---|---|---|
| `tests/functional/test_real_api_integration.py` | 服务连接检查 | 实际外部 API |
| `tests/functional/test_complete_arc_easy_dataset.py` | ARC-Easy 批量比较 | 实际外部 API |
| `tests/functional/test_complete_dataset_evaluation.py` | 开发时的流程和负载实验 | 模拟响应 |
| `scripts/merge_arc_easy_results.py` | 按样本 ID 合并历史结果 | 本地结果文件 |

在仓库根目录使用表中完整路径。实际 API 运行需要环境变量中的密钥、明确的样本范围和预算；运行结果应保存为新的证据，不能覆盖历史报告。数据下载和完整应用启动方法见 [Makefile](Makefile) 。

## 当前维护边界

平台目录包含尚未整体验证的 API、任务、监控和界面代码。离线检查通过不代表完整平台可用。共享 Redis 缓存已使用 JSON，旧 pickle 值按无效缓存处理；需要重新生成缓存。

[MIT License](LICENSE)

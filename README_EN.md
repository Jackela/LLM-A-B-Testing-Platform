# LLM A/B Testing Platform

This repository keeps experiments for comparing language-model responses, statistical tools and an early platform scaffold. AI maintains the introduction and repository.

[中文](README.md)

## Start here

- [Experiment provenance](experiments/evidence.json) binds source hashes, reported counts and runner modes.
- The [2025 ARC-Easy report](ARC_EASY_COMPLETE_TEST_REPORT.md) reports 2,088 of 5,197 samples, about 40.2%. Raw results are absent from Git and the run has not been independently reproduced. Its claims about complete validation, production readiness and model rankings are historical statements, not current findings.
- [Statistical methods](src/domain/analytics/entities/statistical_test.py) are implemented. Bounded offline checks compare independent, paired and Welch t-tests with SciPy and verify insufficient-sample, unequal-pair and non-finite-input failures.
- [AGENTS.md](AGENTS.md) describes maintenance sources and checks.

## Offline checks

Use Python 3.11 and an isolated environment:

```bash
python -m venv .venv-offline
.venv-offline/bin/python -m pip install -r requirements-offline.txt
.venv-offline/bin/python tools/verify_experiment_evidence.py
.venv-offline/bin/python -m unittest discover -s tests/offline -v
```

These component checks cover statistics, provenance, shared-cache serialization, malformed-token rejection and dependency-audit failures. They do not start or validate the full platform or call model services. Poetry manages the full application; its existing 80% test-coverage requirement remains.

CI checks these contracts and audits all applicable locked runtime dependencies on Python 3.11 and 3.12. Missing scanners, malformed output and tool failures fail the check. Use `poetry check --lock` to verify the lock and `make security-scan` for source and dependency checks.

## Experiment entry points

| File | Purpose | Mode |
|---|---|---|
| `tests/functional/test_real_api_integration.py` | Connectivity checks | Real external API |
| `tests/functional/test_complete_arc_easy_dataset.py` | Batch ARC-Easy comparison | Real external API |
| `tests/functional/test_complete_dataset_evaluation.py` | Development flow and load experiments | Simulated responses |
| `scripts/merge_arc_easy_results.py` | Merge historical results by sample ID | Local result files |

Run these paths from the repository root. Real-provider runs need environment credentials, a defined sample range and a budget. Record each new run separately. See [Makefile](Makefile) for application commands.

The API, task, monitoring and UI scaffold has not been validated as a complete application. Shared Redis values now use JSON; old pickle entries become cache misses and must be regenerated.

[MIT License](LICENSE)

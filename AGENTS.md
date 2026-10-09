# Maintenance entry

This repository contains language-model comparison experiments and an unfinished platform scaffold. Maintenance is performed by AI; current capabilities must be checked against source and actual results.

- Use Markdown for explanations and JSON for experiment provenance. Keep existing licenses and the project version.
- Read `README.md`, `pyproject.toml`, `experiments/evidence.json` and the affected module before editing.
- The 2025 ARC-Easy report is historical, partial and not independently reproduced from committed raw results. Keep its original bytes; record new runs separately with their actual inputs and outputs.
- Run `python tools/verify_experiment_evidence.py` and `python -m unittest discover -s tests/offline -v` in the bounded offline environment for these contracts. These checks do not establish that the entire platform works.
- Full application tests retain the existing 80% coverage requirement in `pyproject.toml`. Real-provider scripts under `tests/functional/` require separate credentials and an explicit run budget; ordinary maintenance checks must not call them.
- Prefer small changes to existing modules. Verify both successful behavior and meaningful failure cases. Never turn a failing check into a passing status by discarding its exit code.

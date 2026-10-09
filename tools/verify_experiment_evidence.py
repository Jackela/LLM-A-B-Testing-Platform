"""Check committed experiment provenance without calling providers or loading data."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/evidence.json"


def verify(root: Path = ROOT, record: dict | None = None) -> None:
    record = record or json.loads((root / "experiments/evidence.json").read_text())
    if record["evidence_kind"] != "historical_report_only":
        raise ValueError("The historical report cannot establish a reproduced run")
    for item in record["sources"]:
        path = root / item["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Source changed: {item['path']}")
    historical = record["reported_run"]
    if historical["completed_samples"] != sum(historical["outcomes"].values()):
        raise ValueError("Reported outcome counts do not equal completed samples")
    percentage = 100 * historical["completed_samples"] / historical["planned_samples"]
    if round(percentage, 1) != historical["completion_percent"]:
        raise ValueError("Reported completion percentage does not match counts")
    tracked = subprocess.check_output(["git", "ls-files", "-z", "logs"], cwd=root)
    if tracked or record["raw_results_committed"] is not False:
        raise ValueError("Reassess evidence status when raw results are committed")
    for item in record["runners"]:
        if not (root / item["path"]).is_file() or item["mode"] not in {"real_api", "simulated"}:
            raise ValueError(f"Invalid runner: {item}")


if __name__ == "__main__":
    verify()
    print("Historical source hashes and reported counts agree; raw-run reproduction is unverified.")

"""Export all locked runtime packages, markers and hashes for an audit."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def export() -> str:
    lock = tomllib.loads((ROOT / "poetry.lock").read_text())
    lines = []
    for package in lock["package"]:
        if "main" not in package.get("groups", []):
            continue
        marker = package.get("markers")
        if isinstance(marker, dict):
            marker = marker.get("main")
        requirement = f"{package['name']}=={package['version']}"
        if marker:
            requirement += f" ; {marker}"
        hashes = [item["hash"] for item in package["files"]]
        if not hashes or not all(item.startswith("sha256:") for item in hashes):
            raise ValueError(f"Missing source hashes: {package['name']}")
        lines.append(
            requirement + " \\\n    " + " \\\n    ".join(f"--hash={value}" for value in hashes)
        )
    if not lines:
        raise ValueError("Lock file has no runtime packages")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(export(), end="")

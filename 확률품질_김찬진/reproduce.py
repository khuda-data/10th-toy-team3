"""Run the complete probability-quality reproduction pipeline from raw data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKSUM_FILE = PROJECT_ROOT / "expected" / "input_checksums.json"
PIPELINE_CONFIG = PROJECT_ROOT / "reproduction_config.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_python() -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            "Exact reproduction requires 64-bit CPython 3.12.x; "
            f"current interpreter is {sys.version.split()[0]}."
        )
    if sys.maxsize <= 2**32:
        raise RuntimeError("A 64-bit Python interpreter is required.")


def validate_inputs() -> None:
    manifest = json.loads(CHECKSUM_FILE.read_text(encoding="utf-8"))
    failures: list[str] = []
    for relative, expected in manifest["files"].items():
        path = PROJECT_ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"checksum mismatch: {relative}\n  expected {expected}\n  actual   {actual}")
    if failures:
        raise RuntimeError("Input validation failed:\n" + "\n".join(failures))
    print(f"Input validation passed ({len(manifest['files'])} files).")


def ensure_fresh_workspace() -> None:
    generated_sentinels = (
        "data/interim/seoul_entries.csv.gz",
        "artifacts/models/m1_logistic.joblib",
        "data/manifests/pre_final_test_freeze.json",
        "reports/experiments/stage_27_candidate_freeze.json",
    )
    existing = [p for p in generated_sentinels if (PROJECT_ROOT / p).exists()]
    if existing:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise RuntimeError(
            "Generated outputs already exist. Reproduction is intentionally one-shot "
            "because the Final Test is sealed. Start from a fresh copy of this folder.\n"
            + formatted
        )


def run_module(module: str, environment: dict[str, str]) -> None:
    command = [sys.executable, "-m", module]
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="validate the five immutable input files and exit",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="run the pipeline without the final unittest suite",
    )
    args = parser.parse_args()

    validate_inputs()
    if args.check_inputs:
        return 0

    validate_python()
    ensure_fresh_workspace()
    config = json.loads(PIPELINE_CONFIG.read_text(encoding="utf-8"))
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONPATH"] = str(PROJECT_ROOT)

    started = time.perf_counter()
    for module in config["pipeline_modules"]:
        run_module(module, environment)

    if not args.skip_tests:
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
        print(f"\n>>> {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)

    elapsed = time.perf_counter() - started
    print(f"\nReproduction completed successfully in {elapsed / 60:.1f} minutes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


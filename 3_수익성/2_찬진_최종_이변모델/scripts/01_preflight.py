from __future__ import annotations

import sys
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_ROOT))

from src.config import TABLE_DIR, ensure_output_dirs  # noqa: E402
from src.data import preflight_report, write_json  # noqa: E402


def main() -> None:
    ensure_output_dirs()
    report = preflight_report()
    output = TABLE_DIR / "preflight_report.json"
    write_json(report, output)
    print(f"[OK] preflight passed: {output}")
    for fold, info in report["folds"].items():
        print(f"  {fold}: rows={info['rows']}, races={info['race_count']}, subsets={info['subsets']}")
    print(f"  labels(train/valid): {report['label_validation']}")
    print("  test outcomes opened: False")


if __name__ == "__main__":
    main()

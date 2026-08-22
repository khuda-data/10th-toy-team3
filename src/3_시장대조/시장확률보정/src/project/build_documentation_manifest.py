"""Checksum the stage-19 documentation, test suite, and test entrypoint."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.data.load_raw import PROJECT_ROOT
from src.data.validate_schema import sha256_file


OUTPUT = PROJECT_ROOT / "data" / "manifests" / "documentation_manifest.json"
STATIC_FILES = (
    "README.md",
    "SETUP.md",
    "TESTING.md",
    "PROJECT_GUIDELINES.md",
    "requirements.txt",
    "scripts/run_tests.ps1",
    "reports/experiments/stage_19_summary.md",
)


def main() -> int:
    relative_files = list(STATIC_FILES)
    relative_files.extend(
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        for path in sorted((PROJECT_ROOT / "tests").glob("test_*.py"))
    )
    files = []
    for relative in relative_files:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    manifest = {
        "manifest_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "stage 19 documentation and regression tests",
        "test_framework": "unittest discovery",
        "test_command": "python -m unittest discover -s tests -v",
        "test_module_count": len(list((PROJECT_ROOT / "tests").glob("test_*.py"))),
        "files": files,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

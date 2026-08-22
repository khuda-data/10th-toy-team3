"""Build a small historical-only fixture demonstrating the prediction contract."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.data.load_raw import PROJECT_ROOT
from src.data.model_data import load_model_entries
from src.data.validate_schema import sha256_file
from src.inference.prediction_contract import generate_predictions, write_schema_manifest


OUTPUT = PROJECT_ROOT / "data" / "predictions" / "stage_18_contract_fixture.csv"
REPORT = PROJECT_ROOT / "reports" / "experiments" / "stage_18_prediction_contract.json"


def main() -> int:
    test = load_model_entries(("test",))
    race_ids = test["race_id"].drop_duplicates().head(2)
    fixture = test.loc[test["race_id"].isin(race_ids)].copy()
    fixture["winOdds_snapshot"] = fixture["winOdds"]
    fixture["odds_source"] = "historical_closing_odds_fixture_not_live"
    for race_id, indices in fixture.groupby("race_id").groups.items():
        race_date = pd.to_datetime(str(fixture.loc[indices, "rcDate"].iloc[0]))
        start = race_date.tz_localize("Asia/Seoul") + pd.Timedelta(hours=12)
        fixture.loc[indices, "odds_snapshot_time"] = (start - pd.Timedelta(minutes=1)).isoformat()
        fixture.loc[indices, "prediction_time"] = (start - pd.Timedelta(seconds=30)).isoformat()
        fixture.loc[indices, "race_start_time"] = start.isoformat()

    output = generate_predictions(fixture)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    schema = write_schema_manifest()
    report = {
        "experiment": "stage_18_prediction_output_contract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "schema": "data/manifests/prediction_output_schema.json",
        "fixture": {
            "path": "data/predictions/stage_18_contract_fixture.csv",
            "rows": int(len(output)),
            "races": int(output["race_id"].nunique()),
            "sha256": sha256_file(OUTPUT),
            "purpose": "Schema demonstration only; timestamps are synthetic and odds are historical closing odds.",
        },
        "model_version": schema["model_version"],
        "valid_actions": output["action"].value_counts().to_dict(),
        "live_readiness": False,
        "blocking_live_requirements": [
            "real pre-race odds snapshots with timestamps",
            "authoritative race start times and full entry lists",
            "monitoring for schema drift and rejected races",
        ],
        "final_test_model_changed": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

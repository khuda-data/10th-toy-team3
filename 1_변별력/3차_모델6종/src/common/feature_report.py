# -*- coding: utf-8 -*-
"""stub — 원본 src/common/feature_report 를 대체한다.

중요도 오름차순 하위 70개를 CSV 로 쓴다.
"""
import csv


def write_bottom_70(features, importance, path, kind="importance"):
    rows = sorted(zip(features, [float(x) for x in importance]), key=lambda r: r[1])[:70]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["feature", kind])
        w.writerows(rows)
    return path

"""
V11.2_ALT_iteration_comparison.py — compare the ALT iterations' headline results
from archived evidence and verify the champion claim:
  best validated results = V11.2 dossier; underlying model = frozen V10.5.3 ridge.

Hard-asserts (abort on failure — packaging is gated on this script):
  A1  version_comparison.csv ranks V10.5.3 first (AUROC 0.9267)
  A2  V10.5.3 ridge spec lovo_auroc == 0.9267, threshold_youden == 0.4456
  A3  V11.2 metric suite: auroc == 0.9267, threshold == 0.4456, and
      _meta.source_csv points at the V10.5.3 predictions CSV
      (proves V11.2 'model' IS the frozen V10.5.3 ridge — no new model in V11.2)

Outputs -> V11.2_ALT/models/V10.5.3_ridge_frozen_champion/:
  ALT_last5_iteration_comparison.csv
  ALT_last5_iteration_comparison.md

Run: py -3 "V11.2_ALT/src/V11.2_ALT_iteration_comparison.py"
"""
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(r"D:\Daimler-starter_motor_alternator_battery")
VC_CSV = ROOT / "V5.2_ALT" / "results" / "V10.5.3_20_5_ALT_version_comparison.csv"
SPEC = ROOT / "V5.2_ALT" / "models" / "classification" / "V10.5.3_20_5_ALT_ridge_spec.json"
SUITE = ROOT / "V11.2_ALT" / "results" / "V11.2_ALT_metric_suite.json"
W106 = ROOT / "V10.6_ALT" / "results" / "V10.6_ALT_chosen_weights.json"
OUT = ROOT / "V11.2_ALT" / "models" / "V10.5.3_ridge_frozen_champion"


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    vc = pd.read_csv(VC_CSV)
    best = vc.sort_values("auroc", ascending=False).iloc[0]
    assert best["version"] == "V10.5.3" and abs(best["auroc"] - 0.9267) < 1e-9, \
        f"A1 FAIL: best is {best['version']} {best['auroc']}"

    spec = json.loads(SPEC.read_text())
    assert spec["lovo_auroc"] == 0.9267 and spec["threshold_youden"] == 0.4456, "A2 FAIL"

    suite = json.loads(SUITE.read_text())
    assert suite["auroc"] == 0.9267, f"A3 FAIL: suite auroc {suite['auroc']}"
    assert suite["_meta"]["threshold"] == 0.4456, "A3 FAIL: suite threshold"
    assert "V10.5.3" in suite["_meta"]["source_csv"], \
        "A3 FAIL: V11.2 suite not sourced from V10.5.3 predictions"
    print("[A1-A3] PASS: V11.2 best-validated results == frozen V10.5.3 ridge (0.9267)")

    w106 = json.loads(W106.read_text())

    rows = [
        {"iteration": "V5.2 Ridge", "headline_auroc": 0.907, "role": "runner-up model",
         "notes": "17-feature ridge",
         "evidence": "V5.2_ALT/results/V10.5.3_20_5_ALT_version_comparison.csv"},
        {"iteration": "V5.2.1", "headline_auroc": 0.887, "role": "superseded",
         "notes": "11-feature production ridge, spec 1.00",
         "evidence": "V5.2_ALT/results/V10.5.3_20_5_ALT_version_comparison.csv"},
        {"iteration": "V10.5.3", "headline_auroc": 0.9267, "role": "CHAMPION MODEL (frozen)",
         "notes": "6-feature ridge, threshold 0.4456; frozen 2026-06-01",
         "evidence": "V5.2_ALT/models/classification/V10.5.3_20_5_ALT_ridge_spec.json"},
        {"iteration": "V10.6.x", "headline_auroc": None, "role": "alert layer on frozen model",
         "notes": (f"ridge_w={w106['ridge_w']}, rules_w={w106['rules_w']}, "
                   f"red recall {w106['red_recall_of_6']} — no new classifier"),
         "evidence": "V10.6_ALT/results/V10.6_ALT_chosen_weights.json"},
        {"iteration": "V11.1", "headline_auroc": 0.9267, "role": "froze the model",
         "notes": "leadership deck; model unchanged",
         "evidence": "V11.1_ALT/ (deck + cache reuse V10.5.3 predictions)"},
        {"iteration": "V11.2", "headline_auroc": 0.9267, "role": "BEST VALIDATED RESULTS",
         "notes": "validation dossier of the frozen model: 139/150 concordant, "
                  "PR-AUC 0.94; NO new model trained",
         "evidence": "V11.2_ALT/results/V11.2_ALT_metric_suite.json"},
        {"iteration": "V12 (GED)", "headline_auroc": 0.9267, "role": "challenge failed",
         "notes": "all 7 GED features REJECTED vs frozen 0.9267 (best delta -0.0067); "
                  "V12.2 exact repro 25/25",
         "evidence": "V12_ALT_GED/results/4_lovo_trial.csv"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ALT_last5_iteration_comparison.csv", index=False)

    md = ["# ALT — last-iterations comparison and champion verification",
          f"_Generated {date.today().isoformat()} by V11.2_ALT_iteration_comparison.py_", "",
          "**VERDICT:** user claim \"V11.2 produced best results\" — **VERIFIED, with lineage nuance**: "
          "V11.2 is the deepest validation of the champion (AUROC 0.9267, 139/150 pairs, "
          "threshold 0.4456) but trained no model; the underlying artifact is the "
          "**V10.5.3 ridge, frozen 2026-06-01** and unbeaten through V10.6.x/V11.x/V12.", "",
          df.to_markdown(index=False)]
    (OUT / "ALT_last5_iteration_comparison.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote comparison table -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

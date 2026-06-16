"""
V10.6.2 Alternator — Per-VIN narratives  (plan W5b)
====================================================
One honest sentence per VIN combining the validated signals (risk tier,
RUL band, GED emergency).  Output: results/V10.6.2_ALT_narratives.json
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V10_6_2_ALT_config", "V10.6.2_ALT_config.py")


def main() -> None:
    pred = pd.read_csv(pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_rul_predictions.csv")
    out = {}
    for _, r in pred.iterrows():
        vin = r["vin_label"]
        if r["failed_flag"] == 1:
            lead = r["would_have_flagged_lead_days"]
            ged_txt = (f"GED=2 would have flagged it {lead}d early"
                       if str(lead) not in ("no_precursor", "", "nan") else "no GED=2 precursor")
            out[vin] = {
                "headline": f"{vin}: FAILED (in dataset) | {ged_txt}",
                "risk_tier": "N/A (failed)",
                "recommendation": r["recommendation"],
            }
        else:
            band = f"{r['rul_p10_days']:.0f}-{r['rul_p90_days']:.0f}d (median {r['median_rul_days']:.0f}d)"
            emer = " | GED=2 EMERGENCY" if r["ged_emergency"] else ""
            out[vin] = {
                "headline": (f"{vin}: {r['risk_tier']} (ridge {r['ridge_prob']:.2f}, "
                             f"{r['ridge_band']}) | RUL band {band}{emer}"),
                "risk_tier": r["risk_tier"],
                "risk_confidence": r["risk_confidence"],
                "timing_confidence": r["timing_confidence"],
                "rul_band_days": band,
                "recommendation": r["recommendation"],
            }
    p = pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_narratives.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[narrative_rul] Saved narratives for {len(out)} VINs")


if __name__ == "__main__":
    main()

"""
V11.1_ALT — Per-VIN narratives  (Task 7)
=========================================
Generates one structured narrative dict per VIN, saved to:
  results/V11.1_ALT_narratives.json

Every narrative MUST state:
  1. The fleet window framing (empirical median_ttf_days = 601d).
  2. The truck's RUL band (p10–p90) and median.
  3. Risk band (green / amber / red).
  4. Emergency state.
  5. Because chosen_variant == "M0" (NO_IMPROVEMENT), include the sentence:
     "Covariate channels did not improve per-truck timing (NO_IMPROVEMENT);
      the fleet wear-out curve is shown."

Additional conditional sentences:
  - early-watch trucks: "Compound electrical-distress signature currently active."
  - GED trucks:         "GED=2 excitation storm was observed (emergency channel)."
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import pandas as pd

_src = pathlib.Path(__file__).resolve().parent


def _load(mod_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(mod_name, str(_src / file_name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfg = _load("V11_1_ALT_config", "V11_1_ALT_config.py")

# Read-only backtest reference for chosen_variant
_BACKTEST_JSON = pathlib.Path(cfg.BACKTEST_CACHE) / "backtest_results.json"

# Standard sentence required for all non-failed narratives
_NO_IMPROVE_SENTENCE = (
    "Covariate channels did not improve per-truck timing (NO_IMPROVEMENT); "
    "the fleet wear-out curve is shown."
)


def _build_narrative(r: pd.Series, fleet_median: float, chosen_variant: str) -> dict:
    vin = r["vin_label"]
    failed = int(r["failed_flag"]) == 1

    if failed:
        ged_txt = (
            "GED=2 excitation storm was observed (emergency channel)."
            if r.get("ged_fired", False) else "No GED=2 storm observed."
        )
        cmpd_txt = (
            "Compound electrical-distress signature was active at last known timestamp."
            if int(r.get("early_watch_current", 0) or 0) == 1 else ""
        )
        body = (
            f"{vin} is a FAILED truck (historical record). "
            f"Fleet wear-out window: empirical median TTF = {fleet_median:.0f}d. "
            f"{ged_txt}"
        )
        if cmpd_txt:
            body += f" {cmpd_txt}"
        body += f" {_NO_IMPROVE_SENTENCE}"
        return {
            "headline": f"{vin}: FAILED (historical record)",
            "narrative": body,
            "risk_band": "N/A",
            "emergency_state": str(r.get("emergency_state", "")),
            "recommendation": str(r.get("recommendation", "")),
            "variant": chosen_variant,
        }

    # Non-failed
    p10 = float(r["rul_p10_days"])
    p90 = float(r["rul_p90_days"])
    med = float(r["median_rul_days"])
    rb = str(r["risk_band"])
    above = bool(r["above_thr"])
    time_dim = str(r["time_dim"])
    es = str(r["emergency_state"])
    ridge_prob = float(r["ridge_prob"])

    rul_band_txt = f"{p10:.0f}–{p90:.0f}d (median {med:.0f}d)"

    body = (
        f"{vin}: Fleet wear-out window = {fleet_median:.0f}d empirical median. "
        f"RUL band (80% PI, survival-conditioned) = {rul_band_txt}. "
        f"Risk band = {rb.upper()} (ridge score {ridge_prob:.3f}, "
        f"{'above' if above else 'below'} decision threshold {cfg.RIDGE_DECISION_THR}). "
        f"Time dimension = {time_dim}. "
        f"Emergency state = {es}. "
        f"{_NO_IMPROVE_SENTENCE}"
    )

    # Conditional sentences
    if es == "early-watch":
        body += " Compound electrical-distress signature currently active."
    if es == "emergency" or bool(r.get("ged_fired", False)):
        body += " GED=2 excitation storm was observed (emergency channel)."

    return {
        "headline": (
            f"{vin}: {rb.upper()} risk | RUL {rul_band_txt} | {es}"
        ),
        "narrative": body,
        "risk_band": rb,
        "above_thr": above,
        "time_dim": time_dim,
        "emergency_state": es,
        "rul_p10_days": p10,
        "rul_p90_days": p90,
        "median_rul_days": med,
        "fleet_window_median_days": fleet_median,
        "recommendation": str(r.get("recommendation", "")),
        "variant": chosen_variant,
    }


def main() -> None:
    results_dir = pathlib.Path(cfg.RESULTS_DIR_V11_1)
    results_dir.mkdir(parents=True, exist_ok=True)

    final = pd.read_csv(pathlib.Path(cfg.RUL_CACHE) / "final_rul_per_vin.csv")

    with open(_BACKTEST_JSON) as fh:
        bt = json.load(fh)
    chosen_variant = bt.get("chosen_variant", "M0")

    fleet_median = float(final["fleet_window_median_days"].iloc[0])

    out = {}
    for _, r in final.iterrows():
        vin = r["vin_label"]
        out[vin] = _build_narrative(r, fleet_median, chosen_variant)

    p = results_dir / "V11.1_ALT_narratives.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[narrative_rul] Saved V11.1_ALT_narratives.json  ({len(out)} VINs)")

    # Print 3 sample narratives as requested
    # 1. early-watch NF (likely none; fallback to any NF)
    nf = final[final["failed_flag"] == 0]
    ew_nf = nf[nf["early_watch_current"] == 1]
    if len(ew_nf) > 0:
        sample1_vin = ew_nf.iloc[0]["vin_label"]
        sample1_label = "early-watch NF"
    else:
        # No early-watch NF; pick any NF (lowest ridge to be representative)
        sample1_vin = nf.sort_values("ridge_prob").iloc[0]["vin_label"]
        sample1_label = "any NF (no early-watch NF exists)"

    # 2. VIN1_F_ALT
    sample2_vin = "VIN1_F_ALT"

    # 3. red-band NF (if any, else highest-ridge NF)
    red_nf = nf[nf["risk_band"] == "red"]
    if len(red_nf) > 0:
        sample3_vin = red_nf.sort_values("ridge_prob", ascending=False).iloc[0]["vin_label"]
        sample3_label = "red-band NF"
    else:
        sample3_vin = nf.sort_values("ridge_prob", ascending=False).iloc[0]["vin_label"]
        sample3_label = "highest-ridge NF (no red-band NF)"

    print("\n" + "=" * 70)
    print(f"SAMPLE 1 — {sample1_label}: {sample1_vin}")
    print("=" * 70)
    print(json.dumps(out[sample1_vin], indent=2))

    print("\n" + "=" * 70)
    print(f"SAMPLE 2 — VIN1_F_ALT (failed): {sample2_vin}")
    print("=" * 70)
    print(json.dumps(out[sample2_vin], indent=2))

    print("\n" + "=" * 70)
    print(f"SAMPLE 3 — {sample3_label}: {sample3_vin}")
    print("=" * 70)
    print(json.dumps(out[sample3_vin], indent=2))


if __name__ == "__main__":
    main()

"""
V10.6.2 Alternator — Customer Markdown Report  (plan W5b)
=========================================================
Written fresh (not adapted) so it structurally avoids the two V10.6.1 bugs:
  B1  reads median from `median_ttf_days` (now present in params JSON).
  B2  reads verification gate statuses by NAME from verification.json.

Deliverable posture (locked decision D2): lead with the three VALIDATED
signals — risk tier, fleet replacement window, GED emergency — and demote the
per-truck RUL band to a caveated appendix with the backtest honesty section.

Output: reports/V10.6.2_ALT_customer_report.md
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


def _md_table(df, cols=None):
    df = df[cols] if cols else df
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def main() -> None:
    wb = json.loads((pathlib.Path(cfg.WEIBULL_CACHE) / "fleet_weibull_params.json").read_text())
    fw = json.loads((pathlib.Path(cfg.RUL_CACHE) / "fleet_window.json").read_text())
    bt = json.loads((pathlib.Path(cfg.BACKTEST_CACHE) / "backtest_results.json").read_text())
    ver_path = pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_verification.json"
    ver = json.loads(ver_path.read_text()) if ver_path.exists() else None

    pred = pd.read_csv(pathlib.Path(cfg.RESULTS_DIR_V2) / "V10.6.2_ALT_rul_predictions.csv")
    ged = pd.read_csv(pathlib.Path(cfg.GED_EMERGENCY_CACHE) / "ged_emergency.csv")

    nf = pred[pred["failed_flag"] == 0].copy()
    rew = bt["time_rewound"]["overall"]
    tot = bt["lovo_total_ttf"]

    L = []
    L.append("# V10.6.2 Alternator — RUL & Risk Report (Honest Baseline)")
    L.append("")
    L.append(f"**Version:** {cfg.VERSION} | **Classifier:** V10.5.3 frozen "
             f"(LOVO AUROC 0.927) | **Fleet:** 25 trucks (10 failed + 15 non-failed)")
    L.append("")
    L.append("## Executive Summary")
    L.append("")
    L.append(f"- **Which trucks (validated):** the frozen classifier ranks failure risk at "
             f"AUROC 0.927. Among the 15 in-service trucks, **{int((nf['risk_tier']=='HIGH_RISK').sum())}** "
             f"is high-risk (ridge ≥ {cfg.RIDGE_DECISION_THR}).")
    L.append(f"- **When (fleet-level):** alternators in this fleet fail in a wear-out band centred at "
             f"**{fw['median_ttf_days']:.0f} days** "
             f"(p25–p75 {fw['p25_ttf_days']:.0f}–{fw['p75_ttf_days']:.0f}d; "
             f"≈ {fw['median_ttf_km_est']:.0f} km / {fw['median_ttf_ehrs_est']:.0f} engine-hours, estimated).")
    L.append(f"- **Per-truck timing — honest limit:** out-of-sample backtest shows the survival RUL "
             f"point estimate does **not** beat a fleet-clock baseline "
             f"(rewound MAE **{rew['mae_model']:.0f}d** vs **{rew['mae_dummyA']:.0f}d**, verdict "
             f"*{bt['time_rewound']['verdict']}*). Its **intervals are well-calibrated** "
             f"({tot['pi_coverage_k_of_n']} coverage). Treat per-truck RUL as a wide band, not a date.")
    L.append(f"- **Emergency signal:** GED=2 excitation-fault monitor fires for "
             f"**{int(ged[(ged.failed_flag==1)&(ged.ever_fired)].shape[0])}/10** failures "
             f"with **0/15** false alarms — actionable for VIN1 (21-day lead) only.")
    L.append("")

    # --- 1. Risk ranking ---------------------------------------------------
    L.append("## 1. Risk Ranking — *Which* trucks (frozen classifier, AUROC 0.927)")
    L.append("")
    L.append("The validated signal. Prioritise inspections top-down.")
    L.append("")
    rr = nf.sort_values("ridge_prob", ascending=False)[
        ["vin_label", "ridge_prob", "ridge_band", "risk_tier", "recommendation"]
    ].head(8)
    L.append(_md_table(rr))
    L.append("")

    # --- 2. Fleet window ---------------------------------------------------
    L.append("## 2. Fleet Replacement Window — *When* (fleet-level)")
    L.append("")
    L.append(f"- **Empirical anchor (recommended):** median failure at **{fw['median_ttf_days']:.0f} days** "
             f"(p25–p75 {fw['p25_ttf_days']:.0f}–{fw['p75_ttf_days']:.0f}d, "
             f"range {fw['min_ttf_days']:.0f}–{fw['max_ttf_days']:.0f}d), n={fw['n_events']} events.")
    L.append(f"- ≈ **{fw['median_ttf_km_est']:.0f} km** / **{fw['median_ttf_ehrs_est']:.0f} engine-hours** "
             f"(estimated; km is speed-integrated, not odometer).")
    L.append(f"- Survival-model median (context only): **{wb['median_ttf_days']:.0f}d** "
             f"[{wb['ci_lower']:.0f}–{wb['ci_upper']:.0f}], Weibull shape {wb['shape']} (posterior). "
             f"This over-predicts observed failures by ~{round(wb['median_ttf_days']-fw['median_ttf_days'])}d "
             f"due to right-censoring — hence the empirical anchor is used for the headline.")
    L.append("")

    # --- 3. GED emergency --------------------------------------------------
    L.append("## 3. GED=2 Emergency Alerts — *When* (the few real precursors)")
    L.append("")
    L.append(f"Independent event-driven alert (daily GED=2 ≥ {cfg.GED_EMERGENCY_DAILY_COUNT_MIN}/day). "
             f"High precision, low recall — most failures have **no** excitation precursor.")
    L.append("")
    gfire = ged[ged["ever_fired"]][
        ["vin_label", "failed_flag", "n_fire_days", "first_fire_lead_days", "total_ged2_lifetime"]
    ]
    L.append(_md_table(gfire) if len(gfire) else "_No trucks currently firing._")
    L.append("")

    # --- 4. Per-truck RUL band (appendix) ---------------------------------
    L.append("## 4. Per-Truck RUL Band — *secondary, caveated*")
    L.append("")
    L.append("Survival-conditioned predictive interval (80%). The point estimate does **not** beat "
             "the fleet clock (see §5); the **interval** is the trustworthy part. km/engine-hours "
             "are estimated display conversions, not independent accuracy.")
    L.append("")
    ap = nf.sort_values("median_rul_days")[
        ["vin_label", "current_age_days", "median_rul_days", "rul_p10_days", "rul_p90_days",
         "risk_tier", "ged_emergency"]
    ]
    L.append(_md_table(ap))
    L.append("")

    # --- 5. Backtest honesty ----------------------------------------------
    L.append("## 5. Backtest — Honest Validation (out-of-sample, per-fold refit)")
    L.append("")
    L.append(f"- **Total-TTF:** model MAE **{tot['mae_model']:.0f}d** vs fleet-clock dummy "
             f"**{tot['mae_dummyA']:.0f}d**; predictive-interval coverage **{tot['pi_coverage_k_of_n']}** "
             f"(gate ≥ {cfg.PI_COVERAGE_TARGET:.0%}: {'PASS' if tot['pi_gate_pass'] else 'FAIL'}).")
    L.append(f"- **Time-rewound (T-270/-180/-90d):** model MAE **{rew['mae_model']:.0f}d** vs dummy "
             f"**{rew['mae_dummyA']:.0f}d** (signed-rank p={rew['signed_rank_p']}); "
             f"interval coverage {rew['pi_coverage']:.0%}.")
    bh = bt["time_rewound"]["by_horizon"]
    bh_df = pd.DataFrame([
        {"horizon": f"T-{h}d", "model_MAE_d": v["mae_model"], "dummy_MAE_d": v["mae_dummyA"],
         "coverage": v["pi_coverage"]} for h, v in bh.items()
    ])
    L.append("")
    L.append(_md_table(bh_df))
    L.append("")
    L.append(f"**Verdict: {bt['time_rewound']['verdict'].upper()}.** The per-truck survival RUL adds no "
             "day-accuracy over assuming every alternator fails at the fleet median age. This is *why* the "
             "deliverable leads with risk tier + fleet window + GED, not per-truck dates.")
    L.append("")

    # --- 6. Verification ---------------------------------------------------
    L.append("## 6. Verification Gates")
    L.append("")
    if ver:
        vdf = pd.DataFrame([
            {"gate": g["name"], "status": g["status"],
             "type": "gating" if g["gating"] else "documented"}
            for g in ver["gates"]
        ])
        L.append(_md_table(vdf))
        L.append("")
        L.append(f"**Gating: {ver['n_gating_pass']}/{ver['n_gating']} pass — "
                 f"overall {'PASS' if ver['overall_pass'] else 'FAIL'}.**")
    else:
        L.append("_verification.json not found (run verify step)._")
    L.append("")

    # --- 7. Limitations ----------------------------------------------------
    L.append("## 7. Limitations")
    L.append("")
    L.append("1. **n = 10 failure events.** Every survival quantity has wide uncertainty; the fleet "
             "Weibull is unstable enough that an empirical median is a better point estimate.")
    L.append("2. **No per-truck timing signal.** VSI is flat to failure for 8/10 trucks; only GED=2 "
             "(2/10) gives a real precursor. A precise per-truck failure date is not extractable from "
             "the 6 CAN channels.")
    L.append("3. **km / engine-hours are estimated** (speed-integrated, not odometer/hour-meter).")
    L.append("4. **Classifier frozen.** No retraining; ridge_prob is a static whole-life score "
             "(its risk discrimination is borrowed by the RUL layer for ranking, not timing).")
    L.append("")
    L.append("_Generated by V10.6.2 honest-baseline pipeline._")

    out = pathlib.Path(cfg.REPORTS_DIR_V2)
    out.mkdir(parents=True, exist_ok=True)
    (out / "V10.6.2_ALT_customer_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[markdown_report] Saved V10.6.2_ALT_customer_report.md ({len(L)} lines)")


if __name__ == "__main__":
    main()
